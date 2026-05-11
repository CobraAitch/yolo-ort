//! YOLO inference in Rust via ONNX Runtime.
//!
//! Supports YOLOv5 (anchor-based), YOLOv11 (anchor-free), and YOLOv26 (NMS-free)
//! through a single [`Detector`] type with execution-provider fallback
//! (TensorRT -> CUDA -> CPU).
//!
//! ```no_run
//! use yolo_ort::{Detector, DetectorConfig, ModelVariant};
//!
//! let mut detector = Detector::from_file(
//!     "yolo11n.onnx",
//!     ModelVariant::V11,
//!     DetectorConfig::default(),
//! )?;
//! let image = image::open("input.jpg")?.to_rgb8();
//! let detections = detector.detect(&image)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod config;
mod decode;
mod detection;
mod detector;
mod error;
mod model;
mod postprocess;
mod preprocess;
mod session;

pub use config::DetectorConfig;
pub use detection::{BBox, Detection};
pub use detector::Detector;
pub use error::{Result, YoloError};
pub use model::ModelVariant;
pub use session::{ConvAlgorithmSearchPref, ExecutionProviderPreference, SessionOptions};
