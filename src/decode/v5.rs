//! YOLOv5: output `[1, N, 5 + C]`, row = `[cx, cy, w, h, obj, c0..cC]`.
//! Confidence = `obj × max(class_score)`; argmax class wins.

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
    if shape.len() != 3 || shape[0] != 1 || shape[2] < 6 {
        return Err(YoloError::UnexpectedOutputShape {
            variant: "YOLOv5",
            expected: "[1, N, 5+C] with C >= 1",
            got: shape,
        });
    }
    let n = shape[1] as usize;
    let stride = shape[2] as usize;
    let num_classes = stride - 5;

    let raw = view
        .as_slice()
        .ok_or(YoloError::InvalidInput("YOLOv5 output is not contiguous"))?;

    let conf_threshold = config.confidence_threshold;
    let mut candidates: Vec<Detection> = Vec::with_capacity(256);

    for i in 0..n {
        let row_start = i * stride;
        let row = &raw[row_start..row_start + stride];
        let objectness = row[4];
        if objectness < conf_threshold {
            continue;
        }
        let (class_id, class_score) = argmax(&row[5..5 + num_classes]);
        let confidence = objectness * class_score;
        if confidence < conf_threshold {
            continue;
        }
        let bbox = BBox::from_cxcywh(row[0], row[1], row[2], row[3]);
        candidates.push(crate::postprocess::finalise(
            bbox, class_id, confidence, letterbox, config,
        ));
    }
    Ok(candidates)
}

#[inline]
fn argmax(scores: &[f32]) -> (usize, f32) {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if s > best {
            best = s;
            best_idx = i;
        }
    }
    (best_idx, best)
}
