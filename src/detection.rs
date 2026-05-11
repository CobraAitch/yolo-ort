/// Axis-aligned bounding box in source-image pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    #[inline]
    pub const fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    #[inline]
    pub fn from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        let half_w = w * 0.5;
        let half_h = h * 0.5;
        Self {
            x1: cx - half_w,
            y1: cy - half_h,
            x2: cx + half_w,
            y2: cy + half_h,
        }
    }

    #[inline]
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    #[inline]
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    #[inline]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);
        let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }

    #[inline]
    pub(crate) fn clamp_to(&mut self, width: f32, height: f32) {
        self.x1 = self.x1.clamp(0.0, width);
        self.y1 = self.y1.clamp(0.0, height);
        self.x2 = self.x2.clamp(0.0, width);
        self.y2 = self.y2.clamp(0.0, height);
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub class_id: usize,
    pub confidence: f32,
    pub class_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_from_cxcywh_centers_correctly() {
        let b = BBox::from_cxcywh(10.0, 10.0, 4.0, 6.0);
        assert!((b.x1 - 8.0).abs() < 1e-6);
        assert!((b.y1 - 7.0).abs() < 1e-6);
        assert!((b.x2 - 12.0).abs() < 1e-6);
        assert!((b.y2 - 13.0).abs() < 1e-6);
        assert!((b.area() - 24.0).abs() < 1e-6);
    }

    #[test]
    fn iou_identical_boxes_is_one() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        let a = BBox::new(0.0, 0.0, 5.0, 5.0);
        let b = BBox::new(10.0, 10.0, 15.0, 15.0);
        assert!(a.iou(&b) < 1e-6);
    }

    #[test]
    fn iou_half_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 0.0, 15.0, 10.0);
        assert!((a.iou(&b) - (1.0 / 3.0)).abs() < 1e-6);
    }
}
