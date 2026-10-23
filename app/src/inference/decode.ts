/**
 * Raw detector tensor -> candidate boxes, in MODEL-INPUT pixel space. No NMS (nms.ts),
 * no screen-space conversion (toScreen.ts, step 8). Pure: arrays in, objects out --
 * no model, no I/O, no React.
 *
 * SOURCE FORMAT -- determined from evidence, not assumed from the model card:
 *
 *  - Flat layout is CHANNEL-MAJOR: raw[channel * anchors + anchor], i.e. a flattened
 *    [1, features, anchors] tensor. This is a genuine ON-DEVICE read: `model.outputs[0]`
 *    (react-native-fast-tflite's own metadata, queried right after `loadTensorflowModel`
 *    resolves, before any inference call) reports shape [1, 14, 8400] on this device --
 *    features (4 box + 10 class channels) outer, anchors inner.
 *  - Box encoding is cxcywh, normalized to [0, 1] (not xyxy, not 640-px pixel space).
 *    Evidence: channel 0 (cx) over the first 20 anchors of a real output tensor is
 *    [0.0158, 0.0277, 0.0316, ..., 0.2492] -- monotonically increasing, sub-1.0 values
 *    consistent with a normalized x-coordinate scanning the anchor grid left to right.
 *    Whole-tensor min/max was 0.0 / 1.0008 (the sliver over 1.0 is int8-dequant rounding
 *    noise, not evidence of pixel-space coordinates, which would range into the hundreds).
 *    This agrees with docs/model-card-v1.md's own empirical determination (cxcywh at a
 *    100% valid-box rate vs. 0% for xyxy; p95 coordinate 0.89).
 *
 *  HISTORICAL NOTE on where the VALUES above came from: on-device `model.runSync()` used
 *  to throw `TFLite: Failed to run TFLite Model! Status: error` for ANY input, and the
 *  tensor values this decoder was originally written against came from the Python
 *  reference instead. That blocker is resolved as of Lesson 05 step 12 round 2 --
 *  `detector.tflite`'s quantization step (`ai_edge_quantizer`, inside `litert_torch`'s
 *  export path) was serializing large weight buffers with TFLite's out-of-line
 *  `Buffer.Offset()`/`Size()` convention, which the app's pinned `TensorFlowLiteC 2.17.0`
 *  doesn't resolve through either of its model-loading APIs.
 *  `src/smart_scene_analyzer/export/tflite.py`'s `export_detector` now inlines those
 *  buffers back into the classic format as a post-export step, which fixes it. On-device
 *  `runSync()` now succeeds and feeds this decoder real tensors -- see
 *  `docs/parity-evidence/round2-invoke-root-cause.md` for the full diagnosis and fix, and
 *  `app/src/pipeline/analyzeImage.ts`'s header for how it's wired into the real pipeline.
 */
import { labelFor, LABELS, type Label } from '../labels';

/** A box in MODEL-INPUT pixel space: the detector's own padded square canvas (e.g.
 * 640x640), xyxy, absolute pixels. NOT source-image space and NOT screen space --
 * converting to either is toScreen.ts's job (step 8), not this function's. */
export interface ModelInputBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionModelInputPixels {
  box: ModelInputBox;
  classId: number;
  label: Label;
  confidence: number;
}

/** Anchor confidence threshold applied before a candidate is even considered. Matches
 * `CONF_THRESHOLD` in `tests/test_export_parity.py` (the Python reference this module
 * ports) so box counts between the TS and Python paths are directly comparable. */
const CONF_THRESHOLD = 0.25;

export function decodeDetections(
  raw: Float32Array,
  layout: { anchors: number; features: number },
  inputSize: { width: number; height: number }
): DetectionModelInputPixels[] {
  const { anchors, features } = layout;
  const numClasses = features - 4;
  if (numClasses !== LABELS.length) {
    throw new Error(
      `decodeDetections: layout implies ${numClasses} classes but LABELS has ${LABELS.length} -- ` +
        'the layout argument and app/src/labels.ts have drifted apart.'
    );
  }

  const detections: DetectionModelInputPixels[] = [];
  for (let a = 0; a < anchors; a++) {
    let bestClassId = -1;
    let bestScore = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const score = raw[(4 + c) * anchors + a];
      if (score > bestScore) {
        bestScore = score;
        bestClassId = c;
      }
    }
    if (bestScore < CONF_THRESHOLD) continue;

    const cx = raw[0 * anchors + a];
    const cy = raw[1 * anchors + a];
    const w = raw[2 * anchors + a];
    const h = raw[3 * anchors + a];

    detections.push({
      box: {
        x1: (cx - w / 2) * inputSize.width,
        y1: (cy - h / 2) * inputSize.height,
        x2: (cx + w / 2) * inputSize.width,
        y2: (cy + h / 2) * inputSize.height,
      },
      classId: bestClassId,
      label: labelFor(bestClassId),
      confidence: bestScore,
    });
  }
  return detections;
}
