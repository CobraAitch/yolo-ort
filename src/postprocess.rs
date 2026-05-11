use crate::config::DetectorConfig;
use crate::detection::{BBox, Detection};
use crate::preprocess::LetterboxParams;

pub(crate) fn non_max_suppression(
    mut detections: Vec<Detection>,
    iou_threshold: f32,
    max_keep: usize,
) -> Vec<Detection> {
    if detections.is_empty() {
        return detections;
    }
    detections.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; detections.len()];
    let mut kept: Vec<Detection> = Vec::with_capacity(max_keep.min(detections.len()));

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        let pivot = detections[i].clone();
        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }
            if detections[j].class_id != pivot.class_id {
                continue;
            }
            if pivot.bbox.iou(&detections[j].bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
        kept.push(pivot);
        if kept.len() == max_keep {
            break;
        }
    }
    kept
}

pub(crate) fn finalise(
    mut bbox: BBox,
    class_id: usize,
    confidence: f32,
    letterbox: LetterboxParams,
    config: &DetectorConfig,
) -> Detection {
    let (x1, y1) = letterbox.unproject(bbox.x1, bbox.y1);
    let (x2, y2) = letterbox.unproject(bbox.x2, bbox.y2);
    bbox = BBox::new(x1, y1, x2, y2);
    bbox.clamp_to(letterbox.src_size.0 as f32, letterbox.src_size.1 as f32);

    Detection {
        bbox,
        class_id,
        confidence,
        class_name: config.class_name(class_id).map(str::to_owned),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, class: usize, conf: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, y1, x2, y2),
            class_id: class,
            confidence: conf,
            class_name: None,
        }
    }

    #[test]
    fn nms_keeps_highest_confidence_in_overlapping_cluster() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(1.0, 1.0, 11.0, 11.0, 0, 0.8),
            det(50.0, 50.0, 60.0, 60.0, 0, 0.7),
        ];
        let kept = non_max_suppression(dets, 0.5, 100);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].confidence - 0.9).abs() < 1e-6);
        assert!((kept[1].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn nms_does_not_suppress_across_classes() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0, 0.9),
            det(0.0, 0.0, 10.0, 10.0, 1, 0.8),
        ];
        let kept = non_max_suppression(dets, 0.5, 100);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_respects_max_keep() {
        let dets = (0..10)
            .map(|i| {
                let x = (i as f32).mul_add(20.0, 0.0);
                det(x, 0.0, x + 5.0, 5.0, 0, 0.5)
            })
            .collect();
        let kept = non_max_suppression(dets, 0.5, 3);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn finalise_unprojects_into_source_image_space() {
        let config = DetectorConfig::default();
        let letterbox = LetterboxParams {
            scale: 0.5,
            pad_x: 20.0,
            pad_y: 10.0,
            src_size: (800, 600),
        };
        let bbox = BBox::new(40.0, 30.0, 80.0, 70.0);
        let d = finalise(bbox, 5, 0.9, letterbox, &config);
        assert!((d.bbox.x1 - 40.0).abs() < 1e-4);
        assert!((d.bbox.y1 - 40.0).abs() < 1e-4);
        assert!((d.bbox.x2 - 120.0).abs() < 1e-4);
        assert!((d.bbox.y2 - 120.0).abs() < 1e-4);
        assert_eq!(d.class_id, 5);
    }
}
