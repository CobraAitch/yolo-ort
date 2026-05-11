//! Letterbox resize + HWC->CHW f32 normalisation, fused into a single write
//! into the pre-allocated network-input tensor.

use fast_image_resize::images::{Image as FirImage, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::RgbImage;
use ndarray::Array4;

use crate::error::{Result, YoloError};

#[derive(Debug, Clone, Copy)]
pub(crate) struct LetterboxParams {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub src_size: (u32, u32),
}

impl LetterboxParams {
    #[inline]
    pub(crate) fn unproject(&self, x: f32, y: f32) -> (f32, f32) {
        let inv_scale = 1.0 / self.scale;
        ((x - self.pad_x) * inv_scale, (y - self.pad_y) * inv_scale)
    }
}

pub(crate) struct Preprocessor {
    input: Array4<f32>,
    resizer: Resizer,
    target: (u32, u32),
    resize_buf: Vec<u8>,
    last_layout: Option<(u32, u32, u32, u32)>,
}

impl std::fmt::Debug for Preprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preprocessor")
            .field("target", &self.target)
            .field("input_shape", &self.input.shape())
            .field("last_layout", &self.last_layout)
            .finish()
    }
}

const PAD_VALUE: u8 = 114;
const PAD_NORM: f32 = PAD_VALUE as f32 / 255.0;

impl Preprocessor {
    pub(crate) fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(YoloError::InvalidConfig("input dimensions must be non-zero"));
        }
        let mut input = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
        input
            .as_slice_mut()
            .expect("owned Array4 is contiguous")
            .fill(PAD_NORM);
        Ok(Self {
            input,
            resizer: Resizer::new(),
            target: (width, height),
            resize_buf: Vec::new(),
            last_layout: None,
        })
    }

    pub(crate) fn process(&mut self, src: &RgbImage) -> Result<LetterboxParams> {
        let (src_w, src_h) = src.dimensions();
        if src_w == 0 || src_h == 0 {
            return Err(YoloError::InvalidInput("source image has a zero dimension"));
        }
        let (target_w, target_h) = self.target;

        let scale = (target_w as f32 / src_w as f32).min(target_h as f32 / src_h as f32);
        let new_w = ((src_w as f32 * scale).round() as u32).max(1);
        let new_h = ((src_h as f32 * scale).round() as u32).max(1);
        let pad_left = target_w.saturating_sub(new_w) / 2;
        let pad_top = target_h.saturating_sub(new_h) / 2;
        let layout = (new_w, new_h, pad_left, pad_top);

        if self.last_layout != Some(layout) {
            self.input
                .as_slice_mut()
                .expect("owned Array4 is contiguous")
                .fill(PAD_NORM);
            self.last_layout = Some(layout);
        }

        let required = (new_w * new_h * 3) as usize;
        if self.resize_buf.len() < required {
            self.resize_buf.resize(required, 0);
        }

        let src_view = ImageRef::new(src_w, src_h, src.as_raw(), PixelType::U8x3)?;
        let mut dst_view = FirImage::from_slice_u8(
            new_w,
            new_h,
            &mut self.resize_buf[..required],
            PixelType::U8x3,
        )?;
        self.resizer.resize(
            &src_view,
            &mut dst_view,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )?;

        let tensor_slice = self
            .input
            .as_slice_mut()
            .expect("owned Array4 is contiguous");
        hwc_into_chw_at_offset(
            &self.resize_buf[..required],
            new_w as usize,
            new_h as usize,
            tensor_slice,
            target_w as usize,
            target_h as usize,
            pad_left as usize,
            pad_top as usize,
        );

        Ok(LetterboxParams {
            scale,
            pad_x: pad_left as f32,
            pad_y: pad_top as f32,
            src_size: (src_w, src_h),
        })
    }

    pub(crate) fn input_view(&self) -> ndarray::ArrayView4<'_, f32> {
        self.input.view()
    }
}

#[inline]
fn hwc_into_chw_at_offset(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    off_x: usize,
    off_y: usize,
) {
    const INV_255: f32 = 1.0 / 255.0;
    let plane = dst_w * dst_h;
    let (r_plane, g_plane, b_plane) = (0, plane, 2 * plane);

    debug_assert!(off_x + src_w <= dst_w);
    debug_assert!(off_y + src_h <= dst_h);

    for y in 0..src_h {
        let src_row = y * src_w * 3;
        let dst_row = (off_y + y) * dst_w + off_x;
        for x in 0..src_w {
            let s = src_row + x * 3;
            let d = dst_row + x;
            dst[r_plane + d] = f32::from(src[s]) * INV_255;
            dst[g_plane + d] = f32::from(src[s + 1]) * INV_255;
            dst[b_plane + d] = f32::from(src[s + 2]) * INV_255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_norm_is_correct() {
        assert!((PAD_NORM - (114.0 / 255.0)).abs() < 1e-7);
    }

    #[test]
    fn preprocessor_pre_fills_with_pad_value() {
        let p = Preprocessor::new(8, 8).unwrap();
        for &v in p.input.as_slice().unwrap() {
            assert!((v - PAD_NORM).abs() < 1e-7);
        }
    }

    #[test]
    fn process_writes_content_and_leaves_pad() {
        let mut p = Preprocessor::new(8, 4).unwrap();
        let src = RgbImage::from_pixel(4, 4, image::Rgb([255, 255, 255]));
        let params = p.process(&src).unwrap();
        assert!((params.scale - 1.0).abs() < 1e-5);
        assert_eq!((params.pad_x, params.pad_y), (2.0, 0.0));
        let v = p.input_view();
        assert!((v[[0, 0, 0, 0]] - PAD_NORM).abs() < 1e-5);
        assert!((v[[0, 0, 0, 3]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dim_cache_skips_refill_for_same_layout() {
        let mut p = Preprocessor::new(8, 8).unwrap();
        let red = RgbImage::from_pixel(8, 8, image::Rgb([255, 0, 0]));
        p.process(&red).unwrap();
        let layout_after_first = p.last_layout;
        let green = RgbImage::from_pixel(8, 8, image::Rgb([0, 255, 0]));
        p.process(&green).unwrap();
        assert_eq!(p.last_layout, layout_after_first);
        let v = p.input_view();
        assert!((v[[0, 0, 0, 0]] - 0.0).abs() < 1e-5);
        assert!((v[[0, 1, 0, 0]] - 1.0).abs() < 1e-5);
    }
}
