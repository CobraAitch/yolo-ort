//! ```text
//! cargo run --release --example detect -- \
//!     --model yolo11n.onnx --variant v11 --image input.jpg
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use yolo_ort::{
    ConvAlgorithmSearchPref, Detector, DetectorConfig, ExecutionProviderPreference, ModelVariant,
    SessionOptions,
};

#[derive(Parser, Debug)]
#[command(version, about = "Run YOLOv5/v11/v26 inference via ONNX Runtime")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    image: PathBuf,
    #[arg(long, value_enum)]
    variant: VariantArg,
    #[arg(long, default_value_t = 0.25)]
    conf: f32,
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
    #[arg(long, default_value_t = 640)]
    imgsz: u32,
    #[arg(long)]
    cpu: bool,
    #[arg(long, default_value_t = 1)]
    repeat: u32,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long)]
    no_cuda_graphs: bool,
    #[arg(long)]
    no_io_binding: bool,
    #[arg(long)]
    exhaustive_conv_search: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum VariantArg {
    V5,
    V11,
    V26,
}

impl From<VariantArg> for ModelVariant {
    fn from(v: VariantArg) -> Self {
        match v {
            VariantArg::V5 => Self::V5,
            VariantArg::V11 => Self::V11,
            VariantArg::V26 => Self::V26,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let image = image::open(&cli.image)
        .with_context(|| format!("failed to open image: {}", cli.image.display()))?
        .to_rgb8();

    let config = DetectorConfig {
        input_size: (cli.imgsz, cli.imgsz),
        confidence_threshold: cli.conf,
        iou_threshold: cli.iou,
        ..Default::default()
    };
    let session_options = SessionOptions {
        provider_preference: if cli.cpu {
            ExecutionProviderPreference::CpuOnly
        } else {
            ExecutionProviderPreference::Auto
        },
        cuda_graphs: !cli.no_cuda_graphs,
        use_io_binding: !cli.no_io_binding,
        cuda_conv_search: if cli.exhaustive_conv_search {
            ConvAlgorithmSearchPref::Exhaustive
        } else {
            ConvAlgorithmSearchPref::Heuristic
        },
        ..Default::default()
    };

    let mut detector =
        Detector::from_file_with(&cli.model, cli.variant.into(), config, &session_options)
            .with_context(|| format!("failed to load model: {}", cli.model.display()))?;

    detector.warmup(cli.warmup).context("warmup failed")?;

    let mut total_ms = 0.0_f64;
    let mut last = Vec::new();
    for i in 0..cli.repeat.max(1) {
        let start = Instant::now();
        last = detector.detect(&image)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed;
        println!("run {:>3}: {:.2} ms — {} detections", i + 1, elapsed, last.len());
    }
    let mean = total_ms / f64::from(cli.repeat.max(1));
    println!("mean: {mean:.2} ms over {} runs", cli.repeat.max(1));

    println!("\nDetections:");
    for (i, d) in last.iter().enumerate() {
        let label = d.class_name.as_deref().unwrap_or("?");
        println!(
            "  {:>2}. [{:>3}] {:<16} conf={:.3}  box=({:>6.1}, {:>6.1}, {:>6.1}, {:>6.1})",
            i + 1,
            d.class_id,
            label,
            d.confidence,
            d.bbox.x1,
            d.bbox.y1,
            d.bbox.x2,
            d.bbox.y2,
        );
    }
    Ok(())
}
