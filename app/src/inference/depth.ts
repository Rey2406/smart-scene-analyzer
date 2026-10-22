/**
 * Depth Anything V2 depth model loading — react-native-executorch, the 0.10
 * core API (loadModel/tensor/wrapAsync from the package root). Deliberately
 * NOT the /legacy API: no initExecutorch, no resource fetcher.
 *
 * Contract source of truth: docs/model-card-v1.md, "Exported artifact ->
 * Depth -> Depth Anything V2 Small -> depth.pte" (the VERIFIED table, dated
 * 2026-10-19 -- the export that replaced the failed optimum-cli path).
 *
 * Input:  [1, 3, 518, 518] float32 NCHW, contiguous. 518 = 14 x 37, the
 *         ViT-S/14 patch size -- not the detector's 640.
 *         Normalization: ImageNet mean (0.485, 0.456, 0.406),
 *         std (0.229, 0.224, 0.225), per channel.
 * Output: ONE tensor, native resolution [1, 518, 518], unresized.
 *         Semantics: relative inverse depth, larger value is nearer, no
 *         unit, no cross-image scale. Never convert this to a physical
 *         unit and never compare it across two images -- units_guard blocks
 *         the former at the identifier/prose level; the latter is a modeling
 *         error no hook can catch.
 *
 * Loads the model once and caches it. The ExecuTorch core `loadModel` is
 * SYNCHRONOUS -- it compiles the whole graph on the calling thread -- so it
 * is wrapped with `wrapAsync` to run on ExecuTorch's background worklet
 * runtime instead of blocking the JS thread for however long a 24 MiB int8
 * graph takes to compile.
 */
import { Asset } from 'expo-asset';
import {
  loadModel,
  setTelemetryEnabled,
  tensor,
  wrapAsync,
  type DType,
  type Model,
} from 'react-native-executorch';

import type { RGBPixels } from './image';
import { toModelInput, type ChannelNorm } from './preprocess';
import { timeAsync, timeSync } from '../dev/phaseTimers';

/** What docs/model-card-v1.md's VERIFIED table claims for the input tensor. */
const EXPECTED_INPUT_SHAPE = [1, 3, 518, 518] as const;

/** What docs/model-card-v1.md's VERIFIED table claims for the output tensor. */
const EXPECTED_OUTPUT_SHAPE = [1, 518, 518] as const;

/**
 * ImageNet normalization constants, per channel (R, G, B) -- docs/model-card-v1.md's
 * VERIFIED table, not guessed and not re-derived anywhere else. Feeding this model
 * [0,1]-only pixels instead produces a depth map that is smooth, plausible, and wrong.
 */
const DEPTH_NORM: ChannelNorm = {
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
};

export type DepthLoadState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; model: Model; methodName: string; loadMs: number }
  | { status: 'error'; error: unknown };

let state: DepthLoadState = { status: 'idle' };
let inFlight: Promise<DepthLoadState> | null = null;
let telemetryDisabled = false;

/** Explicit "not loaded yet" state for the UI -- loading takes seconds. */
export function getDepthState(): DepthLoadState {
  return state;
}

// loadModel compiles the whole graph synchronously; run it off the JS thread.
const loadModelAsync = wrapAsync(loadModel);

/** Reads a method's declared tensor shapes as plain number[] per parameter. */
function describeShapes(spec: readonly Model['schema'][string]['inputs' | 'outputs'][number][]) {
  return spec.map((param) =>
    param.kind === 'Tensor'
      ? param.shape.map((dim) => (dim.kind === 'constant' ? dim.value : dim.kind))
      : param.kind
  );
}

function shapeMatches(actual: unknown, expected: readonly number[]): boolean {
  return Array.isArray(actual) && actual.length === expected.length && actual.every((d, i) => d === expected[i]);
}

/**
 * Loads depth.pte once and caches the result. Concurrent callers share the
 * same in-flight load; a call after success returns the cached model
 * instantly. Per-inference loading is a bug, not a slow path.
 */
export async function loadDepthModel(): Promise<DepthLoadState> {
  if (!telemetryDisabled) {
    // Nothing about this model's use leaves the device -- disable Software
    // Mansion's anonymous download analytics once, at first load.
    setTelemetryEnabled(false);
    telemetryDisabled = true;
  }

  if (state.status === 'ready') return state;
  if (inFlight) return inFlight;

  state = { status: 'loading' };
  inFlight = (async () => {
    const started = Date.now();
    try {
      // Bundled require() -- app/metro.config.js pushes 'pte' onto
      // resolver.assetExts (verified against a real `expo export` run in
      // Lesson 05 step 4, round 2). expo-asset resolves the require() to a
      // local filesystem path loadModel can open.
      const asset = Asset.fromModule(require('../../assets/models/depth.pte'));
      await asset.downloadAsync();
      const localUri = asset.localUri ?? asset.uri;
      if (!localUri) {
        throw new Error('depth.pte resolved to no local URI after downloadAsync()');
      }
      // ExecuTorch's native loadModel() wants a plain filesystem path, not a URI --
      // passing the 'file://' scheme through verbatim makes it try to open a path
      // that starts with the literal characters "file:", which does not exist, and
      // fails with Error::AccessFailed. Confirmed empirically on-device (Round 2):
      // this was the actual failure before the strip was added.
      const localPath = localUri.startsWith('file://') ? localUri.slice('file://'.length) : localUri;

      const model = await loadModelAsync(localPath);
      const loadMs = Date.now() - started;

      const methodNames = Object.keys(model.schema);
      // The export did not name its method for us; 'forward' is ExecuTorch's
      // own convention, but report what is actually there rather than
      // assuming it.
      const methodName = methodNames.includes('forward') ? 'forward' : methodNames[0];
      if (!methodName) {
        throw new Error(`depth.pte exposes no methods in its schema (found: ${JSON.stringify(methodNames)})`);
      }

      const methodSpec = model.schema[methodName];
      const actualInputs = describeShapes(methodSpec.inputs);
      const actualOutputs = describeShapes(methodSpec.outputs);
      console.log(
        `[depth] cold load ${loadMs}ms -- method="${methodName}" methods=${JSON.stringify(methodNames)} ` +
          `inputs=${JSON.stringify(actualInputs)} outputs=${JSON.stringify(actualOutputs)}`
      );

      if (!shapeMatches(actualInputs[0], EXPECTED_INPUT_SHAPE)) {
        console.warn(
          `[depth] INPUT SHAPE DISAGREEMENT -- model card claimed ${JSON.stringify(EXPECTED_INPUT_SHAPE)}, ` +
            `loaded model's "${methodName}" reports ${JSON.stringify(actualInputs)}. Trust the loaded model, not the card.`
        );
      }
      if (!shapeMatches(actualOutputs[0], EXPECTED_OUTPUT_SHAPE)) {
        console.warn(
          `[depth] OUTPUT SHAPE DISAGREEMENT -- model card claimed ${JSON.stringify(EXPECTED_OUTPUT_SHAPE)}, ` +
            `loaded model's "${methodName}" reports ${JSON.stringify(actualOutputs)}. Trust the loaded model, not the card.`
        );
      }

      state = { status: 'ready', model, methodName, loadMs };
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

// ---------------------------------------------------------------------------
// Round 2 -- the real execute() call.
// ---------------------------------------------------------------------------

/**
 * A relative-inverse-depth grid: the flat output tensor plus the width/height needed
 * to index it as 2D. No unit, no cross-image scale -- see the module header.
 */
export interface RelativeDepthMap {
  /** Row-major, length width * height -- the output tensor's own buffer, unsqueezed
   * out of its leading batch dim. */
  data: Float32Array;
  width: number;
  height: number;
}

export interface DepthInferenceResult {
  /** The OUTPUT tensor's shape, read off the Tensor object straight after execute()
   * returns -- reported verbatim, before any reshape, so a disagreement with
   * EXPECTED_OUTPUT_SHAPE (or with docs/model-card-v1.md) is visible rather than
   * silently absorbed into relativeDepth's width/height. */
  outputShape: readonly number[];
  /** The OUTPUT tensor's dtype, same "verbatim, before interpretation" rule as shape. */
  outputDtype: DType;
  relativeDepth: RelativeDepthMap;
  /** The letterbox transform toModelInput actually used to build the input -- carried
   * through rather than recomputed, so a depth-map pixel can later be mapped back to
   * SOURCE pixels with the exact values that produced it. */
  scale: number;
  padX: number;
  padY: number;
  /** Raw statistics over every element of the output tensor. Relative inverse depth,
   * unitless, comparable only within this one run -- NOT interpreted here as near or
   * far, and NOT a distance in any form. */
  min: number;
  max: number;
  mean: number;
}

type TensorParamSpec = Model['schema'][string]['inputs' | 'outputs'][number];

/** Narrows a method's parameter spec to a Tensor spec, or fails loudly -- a primitive
 * (Int/Bool/...) parameter where a tensor was expected is a real contract break, not
 * something to paper over with a default. */
function tensorSpecOrThrow(spec: TensorParamSpec, label: string) {
  if (spec.kind !== 'Tensor') {
    throw new Error(`[depth] expected ${label} to be a Tensor parameter, got kind='${spec.kind}'`);
  }
  return spec;
}

/** Reads a TensorSpec's shape as plain numbers. depth.pte's shapes are static (the
 * ExecuTorch `MethodMeta` path in schema.ts's module header -- no dynamic range/enum
 * dims), so every dim is expected to be 'constant'; a non-constant dim here means this
 * model needs the dynamic-shape handling this function does not have, and that is a
 * reason to stop, not guess a size. */
function constantShapeOrThrow(spec: ReturnType<typeof tensorSpecOrThrow>, label: string): number[] {
  return spec.shape.map((dim, i) => {
    if (dim.kind !== 'constant') {
      throw new Error(
        `[depth] ${label} dim #${i} is '${dim.kind}', not 'constant' -- dynamic depth ` +
          `shapes are not handled by this function`
      );
    }
    return dim.value;
  });
}

/**
 * Runs depth.pte on already-decoded RGB pixels end to end: letterbox + normalize
 * (preprocess.ts), execute('forward', ...) on the cached model, and reshape the flat
 * output into a 2D relative-depth grid.
 *
 * Deliberately does NOT interpret min/max/mean as near/far -- that is Round 3's job,
 * checked against a real image, not assumed here.
 */
export async function runDepthInference(pixels: RGBPixels): Promise<DepthInferenceResult> {
  const loaded = await loadDepthModel();
  if (loaded.status !== 'ready') {
    throw new Error(`[depth] runDepthInference called before the model was ready (status='${loaded.status}')`);
  }
  const { model, methodName } = loaded;

  const methodSpec = model.schema[methodName];
  const inputSpec = tensorSpecOrThrow(methodSpec.inputs[0]!, `${methodName} input #0`);
  const outputSpec = tensorSpecOrThrow(methodSpec.outputs[0]!, `${methodName} output #0`);
  const inputShape = constantShapeOrThrow(inputSpec, `${methodName} input #0`);
  const outputShape = constantShapeOrThrow(outputSpec, `${methodName} output #0`);

  if (inputShape.length !== 4) {
    throw new Error(
      `[depth] expected a rank-4 [N, C, H, W] input, the loaded model declares ${JSON.stringify(inputShape)}`
    );
  }
  if (inputSpec.dtype !== 'float32') {
    throw new Error(
      `[depth] input tensor dtype is '${inputSpec.dtype}', but preprocess.ts's toModelInput only produces float32`
    );
  }

  const [, , inputH, inputW] = inputShape;
  // Depth's OWN letterbox, timed separately from the detector's (see analyzeImage.ts) --
  // different target size (518 vs 640), different cost. docs/latency-report.md reports
  // this folded into the overall "Preprocess" row, alongside a note on the split.
  const modelInput = timeSync('depthPreprocess', () =>
    toModelInput(pixels, { width: inputW!, height: inputH! }, DEPTH_NORM)
  );

  // The actual execute() call. Synchronous in the core API (model.execute returns
  // immediately with results, not a Promise -- see node_modules/react-native-executorch/
  // src/core/model.ts), so it is wrapped in a 'worklet' function and run through
  // wrapAsync exactly the way react-native-executorch's own bundled task pipelines do
  // (src/extensions/cv/tasks/objectDetection.ts's detectObjectsWorklet), rather than
  // left to block the JS thread for however long a ViT-S/14 forward pass takes.
  const runForwardWorklet = (inputData: Float32Array) => {
    'worklet';
    const inputTensor = tensor('float32', inputShape, inputData);
    const outputTensor = tensor(outputSpec.dtype, outputShape);
    try {
      model.execute(methodName, [inputTensor], [outputTensor]);

      // Report the OUTPUT tensor's shape/dtype VERBATIM, read off the Tensor object
      // itself -- this is Round 2's actual empirical answer, not a repeat of the
      // model-card claim or of EXPECTED_OUTPUT_SHAPE.
      const realShape = outputTensor.shape;
      const realDtype = outputTensor.dtype;

      // Inlined rather than a shared helper: a plain (non-'worklet') function called
      // from inside a 'worklet' closure crosses the react-native-worklets thread
      // boundary as a "remote function" call, which throws synchronously ("[Worklets]
      // Tried to synchronously call a Remote Function") -- hit this for real on-device
      // in Round 2 with a separate `allocateForDtype` helper before inlining it here.
      let scratch: Float32Array | Uint8Array | Int32Array | BigInt64Array;
      switch (realDtype) {
        case 'float32':
          scratch = new Float32Array(outputTensor.numel);
          break;
        case 'int32':
          scratch = new Int32Array(outputTensor.numel);
          break;
        case 'uint8':
        case 'bool':
          scratch = new Uint8Array(outputTensor.numel);
          break;
        case 'int64':
          scratch = new BigInt64Array(outputTensor.numel);
          break;
      }
      const buffer = outputTensor.getData(scratch);
      return { shape: realShape, dtype: realDtype, buffer };
    } finally {
      inputTensor.dispose();
      outputTensor.dispose();
    }
  };

  // THE real, on-device execute() timing -- no preprocessing folded in. This is the
  // number docs/latency-report.md reports as "Depth inference": the one phase of the
  // two models' actual forward passes that genuinely runs on this device right now
  // (detection's Invoke() does not -- see analyzeImage.ts and decode.ts's headers).
  const { shape: outputShapeVerbatim, dtype: outputDtypeVerbatim, buffer } = await timeAsync(
    'depthInference',
    () => wrapAsync(runForwardWorklet)(modelInput.data)
  );

  console.log(
    `[depth] execute('${methodName}') output -- shape=${JSON.stringify(outputShapeVerbatim)} dtype=${outputDtypeVerbatim}`
  );

  // Reshape: drop the leading batch dim to get an explicit (width, height) grid.
  // Handles the documented [1, H, W] case and, defensively, an already-squeezed
  // [H, W] -- anything else is a shape this function does not know how to interpret,
  // so it fails loudly rather than guessing which dims are width/height.
  let mapWidth: number;
  let mapHeight: number;
  if (outputShapeVerbatim.length === 3 && outputShapeVerbatim[0] === 1) {
    mapHeight = outputShapeVerbatim[1]!;
    mapWidth = outputShapeVerbatim[2]!;
  } else if (outputShapeVerbatim.length === 2) {
    mapHeight = outputShapeVerbatim[0]!;
    mapWidth = outputShapeVerbatim[1]!;
  } else {
    throw new Error(
      `[depth] output shape ${JSON.stringify(outputShapeVerbatim)} is neither [1, H, W] nor [H, W] -- ` +
        `don't know how to reshape it into a 2D depth map`
    );
  }

  if (!(buffer instanceof Float32Array)) {
    // The model card and EXPECTED_OUTPUT_SHAPE both describe a float32 output; a
    // different runtime dtype here is new information, not something to silently
    // coerce into min/max/mean arithmetic.
    throw new Error(
      `[depth] output dtype is '${outputDtypeVerbatim}', not 'float32' -- min/max/mean and the ` +
        `2D reshape below both assume a float buffer, so this is reported, not computed, past this point`
    );
  }

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) {
    const v = buffer[i]!;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / buffer.length;

  console.log(`[depth] output stats (relative inverse depth, unitless) -- min=${min} max=${max} mean=${mean}`);

  return {
    outputShape: outputShapeVerbatim,
    outputDtype: outputDtypeVerbatim,
    relativeDepth: { data: buffer, width: mapWidth, height: mapHeight },
    scale: modelInput.scale,
    padX: modelInput.padX,
    padY: modelInput.padY,
    min,
    max,
    mean,
  };
}
