//! Per-variant raw-output decoders.

mod v11;
mod v26;
mod v5;

use ort::session::SessionOutputs;
use ort::value::DynValue;

use crate::config::DetectorConfig;
use crate::detection::Detection;
use crate::error::{Result, YoloError};
use crate::model::ModelVariant;
use crate::preprocess::LetterboxParams;

pub(crate) fn decode(
    variant: ModelVariant,
    outputs: &SessionOutputs<'_>,
    letterbox: LetterboxParams,
    config: &DetectorConfig,
) -> Result<Vec<Detection>> {
    match variant {
        ModelVariant::V5 => v5::decode(outputs, letterbox, config),
        ModelVariant::V11 => v11::decode(outputs, letterbox, config),
        ModelVariant::V26 => v26::decode(outputs, letterbox, config),
    }
}

pub(crate) fn first_output_tensor<'a>(
    outputs: &'a SessionOutputs<'_>,
) -> Result<(Vec<i64>, ndarray::ArrayViewD<'a, f32>)> {
    let value: &DynValue = outputs
        .get("output0")
        .or_else(|| (outputs.len() > 0).then(|| &outputs[0]))
        .ok_or_else(|| YoloError::MissingOutput("output0".into()))?;
    let view = value.try_extract_array::<f32>()?;
    let shape: Vec<i64> = view
        .shape()
        .iter()
        .map(|&d| i64::try_from(d).unwrap_or(i64::MAX))
        .collect();
    Ok((shape, view))
}
