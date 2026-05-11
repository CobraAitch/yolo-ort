//! ONNX Runtime session construction.
//!
//! Registering `TensorRT -> CUDA -> CPU` lets ORT fall back operator-by-operator:
//! if an EP cannot service a node it tries the next one, ultimately landing on
//! CPU. Same code path on any host.

use std::path::Path;

#[cfg(feature = "cuda")]
use ort::execution_providers::{CUDAExecutionProvider, ConvAlgorithmSearch};
#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use ort::session::{Session, builder::GraphOptimizationLevel};

use crate::error::Result;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionProviderPreference {
    #[default]
    Auto,
    CpuOnly,
}

#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub optimization: GraphOptimizationLevel,
    /// `0` = let ORT decide.
    pub intra_threads: usize,
    /// `0` = let ORT decide.
    pub inter_threads: usize,
    pub device_id: i32,
    pub provider_preference: ExecutionProviderPreference,
    /// CUDA Graphs reduce kernel-launch overhead but require static input
    /// shapes. Safe to leave on for fixed-size YOLO inputs.
    pub cuda_graphs: bool,
    pub tensorrt_fp16: bool,
    pub cuda_conv_search: ConvAlgorithmSearchPref,
    pub cuda_tf32: bool,
    /// NHWC convolution layout (faster on Tensor-Core GPUs, may regress on
    /// older hardware).
    pub cuda_prefer_nhwc: bool,
    pub use_io_binding: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConvAlgorithmSearchPref {
    #[default]
    Heuristic,
    /// Benchmarks every algorithm on the first inference. Slow first run,
    /// fastest subsequent runs.
    Exhaustive,
    OrtDefault,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            optimization: GraphOptimizationLevel::Level3,
            intra_threads: 0,
            inter_threads: 0,
            device_id: 0,
            provider_preference: ExecutionProviderPreference::Auto,
            cuda_graphs: true,
            tensorrt_fp16: true,
            cuda_conv_search: ConvAlgorithmSearchPref::Heuristic,
            cuda_tf32: true,
            cuda_prefer_nhwc: false,
            use_io_binding: true,
        }
    }
}

pub(crate) fn build_session(model_path: &Path, options: &SessionOptions) -> Result<Session> {
    let mut builder = Session::builder()?.with_optimization_level(options.optimization)?;

    if options.intra_threads > 0 {
        builder = builder.with_intra_threads(options.intra_threads)?;
    }
    if options.inter_threads > 0 {
        builder = builder.with_inter_threads(options.inter_threads)?;
    }

    let providers = build_provider_chain(options);
    if !providers.is_empty() {
        builder = builder.with_execution_providers(providers)?;
    }

    Ok(builder.commit_from_file(model_path)?)
}

fn build_provider_chain(options: &SessionOptions) -> Vec<ExecutionProviderDispatch> {
    let mut chain: Vec<ExecutionProviderDispatch> = Vec::new();
    if options.provider_preference == ExecutionProviderPreference::Auto {
        #[cfg(feature = "tensorrt")]
        chain.push(tensorrt_provider(options));
        #[cfg(feature = "cuda")]
        chain.push(cuda_provider(options));
    }
    chain.push(CPUExecutionProvider::default().build());
    chain
}

#[cfg(feature = "tensorrt")]
fn tensorrt_provider(options: &SessionOptions) -> ExecutionProviderDispatch {
    let cache_dir = std::env::temp_dir().join("yolo-ort-trt-cache");
    let mut ep = TensorRTExecutionProvider::default()
        .with_device_id(options.device_id)
        .with_engine_cache(true)
        .with_engine_cache_path(cache_dir.display().to_string())
        .with_timing_cache(true);
    if options.tensorrt_fp16 {
        ep = ep.with_fp16(true);
    }
    if options.cuda_graphs {
        ep = ep.with_cuda_graph(true);
    }
    ep.build()
}

#[cfg(feature = "cuda")]
fn cuda_provider(options: &SessionOptions) -> ExecutionProviderDispatch {
    let mut ep = CUDAExecutionProvider::default().with_device_id(options.device_id);
    ep = match options.cuda_conv_search {
        ConvAlgorithmSearchPref::Heuristic => ep.with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic),
        ConvAlgorithmSearchPref::Exhaustive => ep.with_conv_algorithm_search(ConvAlgorithmSearch::Exhaustive),
        ConvAlgorithmSearchPref::OrtDefault => ep,
    };
    if options.cuda_tf32 {
        ep = ep.with_tf32(true);
    }
    if options.cuda_prefer_nhwc {
        ep = ep.with_prefer_nhwc(true);
    }
    if options.cuda_graphs {
        ep = ep.with_cuda_graph(true);
    }
    ep.build()
}
