//! YOLOv11 / YOLOv8 family: output `[1, 4 + C, N]` (channels-first).
//! Each prediction is `[cx, cy, w, h, c0..cC]`; argmax class probability is
//! the confidence (no separate objectness).

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
    if shape.len() != 3 || shape[0] != 1 || shape[1] < 5 {
        return Err(YoloError::UnexpectedOutputShape {
            variant: "YOLOv11",
            expected: "[1, 4+C, N] with C >= 1",
            got: shape,
        });
    }
    let channels = shape[1] as usize;
    let n = shape[2] as usize;
    let num_classes = channels - 4;

    let raw = view
        .as_slice()
        .ok_or(YoloError::InvalidInput("YOLOv11 output is not contiguous"))?;

    let conf_threshold = config.confidence_threshold;
    let mut candidates: Vec<Detection> = Vec::with_capacity(256);

    for i in 0..n {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_class = 0usize;
        for c in 0..num_classes {
            let score = raw[(4 + c) * n + i];
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }
        if best_score < conf_threshold {
            continue;
        }
        let cx = raw[i];
        let cy = raw[n + i];
        let w = raw[2 * n + i];
        let h = raw[3 * n + i];
        let bbox = BBox::from_cxcywh(cx, cy, w, h);
        candidates.push(crate::postprocess::finalise(
            bbox, best_class, best_score, letterbox, config,
        ));
    }
    Ok(candidates)
}
