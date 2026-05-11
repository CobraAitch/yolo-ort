//! YOLOv26: output `[1, N, 6]`, row = `[x1, y1, x2, y2, conf, class_id]`.
//! End-to-end NMS-free thanks to the one-to-one assignment head; we only
//! apply the user-configured confidence threshold.

use ort::session::SessionOutputs;

use crate::config::DetectorConfig;
use crate::decode::first_output_tensor;
use crate::detection::{BBox, Detection};
use crate::error::{Result, YoloError};
use crate::preprocess::LetterboxParams;

pub(super) fn decode(
    outputs: &SessionOutputs<'_>,
    letterbox: LetterboxParams,
    config: &DetectorConfig,
) -> Result<Vec<Detection>> {
    let (shape, view) = first_output_tensor(outputs)?;
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 6 {
        return Err(YoloError::UnexpectedOutputShape {
            variant: "YOLOv26",
            expected: "[1, N, 6]",
            got: shape,
        });
    }
    let n = shape[1] as usize;
    let raw = view
        .as_slice()
        .ok_or(YoloError::InvalidInput("YOLOv26 output is not contiguous"))?;

    let conf_threshold = config.confidence_threshold;
    let mut detections: Vec<Detection> = Vec::with_capacity(n);

    for i in 0..n {
        let row_start = i * 6;
        let conf = raw[row_start + 4];
        if conf < conf_threshold {
            continue;
        }
        let bbox = BBox::new(
            raw[row_start],
            raw[row_start + 1],
            raw[row_start + 2],
            raw[row_start + 3],
        );
        let class_id = raw[row_start + 5] as i64;
        if class_id < 0 {
            continue;
        }
        detections.push(crate::postprocess::finalise(
            bbox,
            class_id as usize,
            conf,
            letterbox,
            config,
        ));
    }
    Ok(detections)
}
