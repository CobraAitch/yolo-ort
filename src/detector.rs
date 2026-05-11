//! High-level [`Detector`] facade.

use std::path::Path;

use image::RgbImage;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session, SessionOutputs};
use ort::value::TensorRef;

use crate::config::DetectorConfig;
use crate::decode::decode;
use crate::detection::Detection;
use crate::error::Result;
use crate::model::ModelVariant;
use crate::postprocess::non_max_suppression;
use crate::preprocess::Preprocessor;
use crate::session::{SessionOptions, build_session};

pub struct Detector {
    session: Session,
    preprocessor: Preprocessor,
    variant: ModelVariant,
    config: DetectorConfig,
    binding: Option<BoundSession>,
}

struct BoundSession {
    binding: IoBinding,
    input_name: String,
}

impl std::fmt::Debug for Detector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Detector")
            .field("variant", &self.variant)
            .field("config", &self.config)
            .field("io_binding", &self.binding.is_some())
            .finish()
    }
}

impl Detector {
    pub fn from_file(
        model_path: impl AsRef<Path>,
        variant: ModelVariant,
        config: DetectorConfig,
    ) -> Result<Self> {
        Self::from_file_with(model_path, variant, config, &SessionOptions::default())
    }

    pub fn from_file_with(
        model_path: impl AsRef<Path>,
        variant: ModelVariant,
        config: DetectorConfig,
        session_options: &SessionOptions,
    ) -> Result<Self> {
        config.validate()?;
        let session = build_session(model_path.as_ref(), session_options)?;
        let (input_w, input_h) = config.input_size;
        let preprocessor = Preprocessor::new(input_w, input_h)?;

        let binding = if session_options.use_io_binding {
            Some(setup_binding(&session)?)
        } else {
            None
        };

        Ok(Self {
            session,
            preprocessor,
            variant,
            config,
            binding,
        })
    }

    pub fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>> {
        let letterbox = self.preprocessor.process(image)?;
        let input_view = self.preprocessor.input_view();
        let input_tensor = TensorRef::from_array_view(input_view)?;

        let raw = if let Some(bound) = self.binding.as_mut() {
            bound.binding.bind_input(&bound.input_name, &input_tensor)?;
            let outputs = self.session.run_binding(&bound.binding)?;
            decode_outputs(&outputs, self.variant, letterbox, &self.config)?
        } else {
            let outputs = self
                .session
                .run(ort::inputs!["images" => input_tensor])?;
            decode_outputs(&outputs, self.variant, letterbox, &self.config)?
        };

        Ok(if self.variant.skips_nms() {
            cap_and_sort(raw, self.config.max_detections)
        } else {
            non_max_suppression(raw, self.config.iou_threshold, self.config.max_detections)
        })
    }

    /// Run `count` inferences on a dummy frame to amortise engine build,
    /// kernel autotune, and CUDA Graph capture before timing measurements.
    pub fn warmup(&mut self, count: usize) -> Result<()> {
        let (w, h) = self.config.input_size;
        let dummy = RgbImage::from_pixel(w, h, image::Rgb([128, 128, 128]));
        for _ in 0..count {
            let _ = self.detect(&dummy)?;
        }
        Ok(())
    }

    pub const fn session(&self) -> &Session {
        &self.session
    }

    pub const fn config(&self) -> &DetectorConfig {
        &self.config
    }

    pub const fn uses_io_binding(&self) -> bool {
        self.binding.is_some()
    }
}

fn decode_outputs(
    outputs: &SessionOutputs<'_>,
    variant: ModelVariant,
    letterbox: crate::preprocess::LetterboxParams,
    config: &DetectorConfig,
) -> Result<Vec<Detection>> {
    decode(variant, outputs, letterbox, config)
}

fn cap_and_sort(mut detections: Vec<Detection>, max_keep: usize) -> Vec<Detection> {
    detections.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    detections.truncate(max_keep);
    detections
}

fn setup_binding(session: &Session) -> Result<BoundSession> {
    let input_name = session
        .inputs()
        .first()
        .map_or_else(|| "images".to_owned(), |o| o.name().to_owned());
    let output_name = session
        .outputs()
        .first()
        .map_or_else(|| "output0".to_owned(), |o| o.name().to_owned());

    let mut binding = session.create_binding()?;
    let cpu_output = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )?;
    binding.bind_output_to_device(&output_name, &cpu_output)?;

    Ok(BoundSession { binding, input_name })
}
