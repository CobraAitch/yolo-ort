use std::sync::Arc;

use crate::error::{Result, YoloError};

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub input_size: (u32, u32),
    pub confidence_threshold: f32,
    /// IoU threshold for NMS. Ignored for v26 (NMS-free).
    pub iou_threshold: f32,
    pub max_detections: usize,
    pub class_names: Option<Arc<[String]>>,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            input_size: (640, 640),
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
            class_names: None,
        }
    }
}

impl DetectorConfig {
    #[inline]
    pub fn class_name(&self, class_id: usize) -> Option<&str> {
        self.class_names
            .as_ref()
            .and_then(|n| n.get(class_id))
            .map(String::as_str)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.input_size.0 == 0 || self.input_size.1 == 0 {
            return Err(YoloError::InvalidConfig("input_size dimensions must be non-zero"));
        }
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err(YoloError::InvalidConfig(
                "confidence_threshold must be within [0, 1]",
            ));
        }
        if !(0.0..=1.0).contains(&self.iou_threshold) {
            return Err(YoloError::InvalidConfig("iou_threshold must be within [0, 1]"));
        }
        Ok(())
    }
}
