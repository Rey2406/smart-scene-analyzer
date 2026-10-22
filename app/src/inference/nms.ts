/**
 * Greedy non-maximum suppression, per class. Space-agnostic (takes and returns boxes
 * in whatever space the caller is in -- this pipeline calls it in MODEL-INPUT pixels,
 * before the step-8 conversion to SOURCE, same as the Python reference:
 * `src/smart_scene_analyzer/inference/nms.py` suppresses in `DET_INPUT`, before
 * `to_source_frame`). No model, no I/O: arrays in, arrays out.
 *
 * **Per class, not global.** A chair overlapping a table is two correct detections;
 * suppressing across classes would delete one of them and the result would look like a
 * recall problem in the model rather than a bug in this function.
 */
import type { DetectionModelInputPixels, ModelInputBox } from './decode';

/** IoU threshold for NMS. Matches `IOU_NMS_THRESHOLD` in
 * `tests/test_export_parity.py` (the Python reference this module ports), so box
 * counts between the TS and Python paths are directly comparable rather than merely
 * similar. `docs/requirements.md` does not state a value itself -- this project's own
 * export-parity test is the recorded assumption (A7 in `inference/nms.py`'s docstring),
 * and this constant exists so the TS and Python sides read it from the same intent
 * rather than drifting independently. */
export const NMS_IOU_THRESHOLD = 0.45;

function area(box: ModelInputBox): number {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

function iou(a: ModelInputBox, b: ModelInputBox): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = area(a) + area(b) - interArea;
  return union <= 0 ? 0 : interArea / union;
}

/**
 * Sort by confidence, keep the best, drop anything of the SAME CLASS that overlaps it
 * past `iouThreshold`, repeat. An empty input returns an empty list, which is a
 * successful result, not an error.
 */
export function nms(
  detections: DetectionModelInputPixels[],
  iouThreshold: number
): DetectionModelInputPixels[] {
  const ordered = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: DetectionModelInputPixels[] = [];

  for (const candidate of ordered) {
    let suppressed = false;
    for (const k of kept) {
      if (k.classId !== candidate.classId) continue; // per class: different classes never suppress each other
      if (iou(k.box, candidate.box) > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push(candidate);
  }

  return kept;
}
