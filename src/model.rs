/// YOLO model variant the [`crate::Detector`] decodes for.
///
/// | Variant | Output shape    | Layout                          | NMS required |
/// |---------|-----------------|---------------------------------|--------------|
/// | `V5`    | `[1, N, 5 + C]` | `[cx, cy, w, h, obj, c0..cC]`   | yes          |
/// | `V11`   | `[1, 4 + C, N]` | `[cx, cy, w, h, c0..cC]` (T)    | yes          |
/// | `V26`   | `[1, N, 6]`     | `[x1, y1, x2, y2, conf, class]` | no (built-in)|
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVariant {
    V5,
    V11,
    V26,
}

impl ModelVariant {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::V5 => "YOLOv5",
            Self::V11 => "YOLOv11",
            Self::V26 => "YOLOv26",
        }
    }

    pub(crate) const fn skips_nms(self) -> bool {
        matches!(self, Self::V26)
    }
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
