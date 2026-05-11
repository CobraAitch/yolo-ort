# yolo-ort

YOLO inference in Rust via [ONNX Runtime](https://onnxruntime.ai/) (using the [`ort`](https://ort.pyke.io/) 2.0 crate).

Supports three model families through a single `Detector` API:

| Variant | ONNX Output      | Layout                          | NMS         |
|---------|------------------|---------------------------------|-------------|
| YOLOv5  | `[1, N, 5+C]`    | `[cx, cy, w, h, obj, c...]`     | external    |
| YOLOv11 | `[1, 4+C, N]`    | `[cx, cy, w, h, c...]` (T)      | external    |
| YOLOv26 | `[1, N, 6]`      | `[x1, y1, x2, y2, conf, class]` | built-in    |

## Features

- **EP fallback chain** — TensorRT -> CUDA -> CPU, gated by Cargo features
- **Zero-copy input** via `TensorRef::from_array_view` (ort 2.0 API)
- **SIMD letterbox** via [`fast_image_resize`](https://github.com/Cykooz/fast_image_resize)
- **Pre-allocated buffers** — no allocation on the inference hot path
- **Per-class NMS** for v5/v11; v26's NMS-free head is honoured
- **Forbidden `unsafe`** in library code, pedantic clippy lints enabled

## Usage

```rust
use yolo_ort::{Detector, DetectorConfig, ModelVariant};

let mut detector = Detector::from_file(
    "yolo11n.onnx",
    ModelVariant::V11,
    DetectorConfig::default(),
)?;
let image = image::open("input.jpg")?.to_rgb8();
for d in detector.detect(&image)? {
    println!("{:?} ({:.2})  {:?}", d.class_name, d.confidence, d.bbox);
}
```

## Cargo features

| Feature              | Effect                                                  |
|----------------------|---------------------------------------------------------|
| `download-binaries`  | (default) Use pyke's prebuilt ORT binaries              |
| `cuda`               | Register CUDA EP (requires system CUDA ≥ 12.8, cuDNN ≥ 9.19) |
| `tensorrt`           | Register TensorRT EP with FP16 + engine cache (implies `cuda`) |
| `directml`           | Register DirectML EP (Windows)                          |
| `coreml`             | Register CoreML EP (macOS)                              |
| `load-dynamic`       | Resolve `libonnxruntime` at runtime instead of linking  |

## CLI example

```bash
cargo run --release --example detect -- \
    --model yolo11n.onnx --variant v11 --image input.jpg
```

For GPU acceleration:

```bash
cargo run --release --features tensorrt --example detect -- \
    --model yolo11n.onnx --variant v11 --image input.jpg --repeat 100
```

## License

Dual-licensed under MIT or Apache-2.0.
