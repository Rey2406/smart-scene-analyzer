/**
 * YOLO11 detection model loading — react-native-fast-tflite.
 *
 * Contract source of truth: docs/model-card-v1.md, "Exported artifact ->
 * Detection -> v1-baseline -> detector.tflite", reconciled against
 * app/assets/models/model_contract.json (the manifest the export step wrote).
 *
 * Input:  [1, 3, 640, 640] float32 NCHW, normalized to [0,1] (divide by 255,
 *         no further scaling). No quantization on the input tensor itself —
 *         `quantize=8` in the export only affects internal weights/activations.
 * Output: ONE tensor, [1, 14, 8400] float32 -- 4 box channels (cxcywh,
 *         normalized to [0,1]) + 10 class-confidence channels, 8400 anchors.
 * Labels: index-for-index against docs/taxonomy.md -- the canonical order lives in
 *         app/src/labels.ts (LABELS), imported from there everywhere a class id needs
 *         a name. Not restated as a second array here.
 *
 * This module loads the model once, caches it, and reports the model's own
 * `inputs`/`outputs` metadata at load time rather than trusting the card --
 * the card can be wrong (it was, twice, before reconciliation); the loaded
 * model cannot.
 */
import {
  loadTensorflowModel,
  type Tensor,
  type TfliteModel,
} from 'react-native-fast-tflite';

/** What docs/model-card-v1.md claims for the input tensor. */
const EXPECTED_INPUT = {
  dataType: 'float32',
  shape: [1, 3, 640, 640],
} as const;

/** What docs/model-card-v1.md claims for the (single) output tensor. */
const EXPECTED_OUTPUTS = [
  {
    dataType: 'float32',
    shape: [1, 14, 8400],
  },
] as const;

export type DetectorLoadState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; model: TfliteModel; loadMs: number }
  | { status: 'error'; error: unknown };

let state: DetectorLoadState = { status: 'idle' };
let inFlight: Promise<DetectorLoadState> | null = null;

/** Explicit "not loaded yet" state for the UI -- loading takes seconds. */
export function getDetectorState(): DetectorLoadState {
  return state;
}

function describe(tensors: readonly Tensor[]) {
  return tensors.map((t) => ({ name: t.name, dataType: t.dataType, shape: t.shape }));
}

function matchesExpected(
  actual: readonly Tensor[],
  expected: readonly { dataType: string; shape: readonly number[] }[]
): boolean {
  if (actual.length !== expected.length) return false;
  return actual.every((tensor, i) => {
    const exp = expected[i];
    return (
      tensor.dataType === exp.dataType &&
      tensor.shape.length === exp.shape.length &&
      tensor.shape.every((dim, j) => dim === exp.shape[j])
    );
  });
}

/**
 * Loads detector.tflite once and caches the result. Safe to call repeatedly
 * (e.g. from multiple screens) -- concurrent callers share the same in-flight
 * load, and a second call after success returns the cached model instantly.
 * Per-inference loading is a bug, not a slow path; this is why the cache
 * exists at module scope rather than inside a render or a capture handler.
 */
export async function loadDetectorModel(): Promise<DetectorLoadState> {
  if (state.status === 'ready') return state;
  if (inFlight) return inFlight;

  state = { status: 'loading' };
  inFlight = (async () => {
    const started = Date.now();
    try {
      // The bundled require() only works because app/metro.config.js pushes
      // 'tflite' onto resolver.assetExts -- see that file's header comment
      // and Lesson 05 step 4, round 2 for how that was verified against a
      // real `expo export` run rather than assumed from the config alone.
      const model = await loadTensorflowModel(
        require('../../assets/models/detector.tflite'),
        [] // default CPU delegate
      );
      const loadMs = Date.now() - started;

      const actualInputs = describe(model.inputs);
      const actualOutputs = describe(model.outputs);
      console.log(
        `[detector] cold load ${loadMs}ms -- inputs=${JSON.stringify(actualInputs)} outputs=${JSON.stringify(actualOutputs)}`
      );

      if (!matchesExpected(model.inputs, [EXPECTED_INPUT])) {
        console.warn(
          `[detector] INPUT SHAPE DISAGREEMENT -- model card claimed ${JSON.stringify(EXPECTED_INPUT)}, ` +
            `loaded model reports ${JSON.stringify(actualInputs)}. Trust the loaded model, not the card.`
        );
      }
      if (!matchesExpected(model.outputs, EXPECTED_OUTPUTS)) {
        console.warn(
          `[detector] OUTPUT SHAPE DISAGREEMENT -- model card claimed ${JSON.stringify(EXPECTED_OUTPUTS)}, ` +
            `loaded model reports ${JSON.stringify(actualOutputs)}. Trust the loaded model, not the card.`
        );
      }

      state = { status: 'ready', model, loadMs };
      return state;
    } catch (error) {
      state = { status: 'error', error };
      throw error;
    } finally {
      inFlight = null;
    }
  })();

  return inFlight;
}
