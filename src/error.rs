use std::path::PathBuf;

pub type Result<T, E = YoloError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum YoloError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("image resize error: {0}")]
    Resize(#[from] fast_image_resize::ResizeError),

    #[error("image buffer error: {0}")]
    ImageBuffer(#[from] fast_image_resize::ImageBufferError),

    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("io error reading {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "unexpected output tensor shape for {variant}: expected {expected}, got {got:?}"
    )]
    UnexpectedOutputShape {
        variant: &'static str,
        expected: &'static str,
        got: Vec<i64>,
    },

    #[error("model output `{0}` not present in session outputs")]
    MissingOutput(String),

    #[error("invalid input image: {0}")]
    InvalidInput(&'static str),

    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
}

impl From<ort::Error<ort::session::builder::SessionBuilder>> for YoloError {
    fn from(err: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        Self::Ort(err.into())
    }
}
