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
 *  IMPORTANT CAVEAT on where the VALUES above came from: on-device `model.runSync()` /
 *  `model.run()` currently throws `TFLite: Failed to run TFLite Model! Status: error` for
 *  ANY input on this device -- a flat all-0.5 tensor fails exactly like the real image
 *  does, so this is a structural failure (invoke(), not decode), not a data problem. The
 *  tensor VALUES cited above (and used to write this decoder) came from running the
 *  identical `app/assets/models/detector.tflite` file through the Python reference
 *  (`ai_edge_litert.Interpreter`, the runtime `tests/test_export_parity.py` already uses)
 *  on the same bundled test image. The SHAPE/DTYPE conclusion is independently confirmed
 *  on-device; the VALUE-level evidence (monotonic cx, min/max) is not, until the runtime
 *  mismatch below is resolved. Full diagnostic in the Round 1 report: interpreter
 *  construction and `TfLiteInterpreterAllocateTensors` succeed on-device (so the graph's
 *  op codes resolve), but `TfLiteInterpreterInvoke` fails generically every time. The
 *  likely cause is a version skew between the export toolchain (`litert-torch` 0.9.4 /
 *  `ai-edge-litert` 2.2.0 -- the newer "LiteRT" generation) and
 *  `react-native-fast-tflite@3.0.1`'s iOS pod, which pins the older classic
 *  `TensorFlowLiteC 2.17.0` (confirmed in `ios/Podfile.lock`; 3.0.1 is also the latest
 *  published version of the library, so there is no newer release to pick up a newer pod).
 *  This is an artifact/toolchain contract issue, not something to route around here --
 *  flagged rather than silently patched, per the mobile agent's constraints.
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
