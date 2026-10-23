/**
 * The ONE pipeline function: image URI in, a ready-to-draw overlay result out. Both
 * Photo mode and Live mode call this SAME function -- neither has its own decoder,
 * letterbox, or depth call. See app/CLAUDE.md's architecture diagram for the named
 * coordinate spaces this composes: SOURCE pixels -> (preprocess.ts's letterbox) ->
 * MODEL-INPUT pixels -> (decode.ts, nms.ts) -> back to SOURCE via toScreen.ts, at draw
 * time, in DetectionOverlay.tsx.
 *
 * Extracted from the Round 3+4 diagnostic screen (git history: App.tsx before step 9) --
 * that screen ran this exact sequence inline, once per bundled test image, to prove the
 * overlay geometry. This module is that sequence, generalized to take any image URI
 * (a picker result, a camera still, or a clip-replay thumbnail) rather than only a
 * `require()`'d asset module.
 *
 * Every stage is real and runs on-device for whatever image `uri` points to:
 * decodeImage, toModelInput (both letterboxes), the detector's own `runSync()` call,
 * decodeDetections, nms, and runDepthInference's `execute()` call. Each kept box's
 * `relativeDepth` is sampled from the real depth map at the box's own region, using the
 * DEPTH model's own (separate) letterbox -- via the real fusion port
 * (`app/src/fusion/index.ts`), not an inline median. See that module's header for the
 * full reduction (median, inset, stride, depth_quality classification) ported from
 * `src/smart_scene_analyzer/fusion.py`.
 *
 * Until Lesson 05 step 12 round 2, the detector half of this was substituted with a
 * frozen Python-reference fixture tensor: `react-native-fast-tflite`'s `Invoke()` failed
 * structurally against this export, on both Simulator and device. Root cause (not a
 * runtime/toolchain mismatch after all): `detector.tflite`'s quantization step
 * (`ai_edge_quantizer`, inside `litert_torch`'s export path) serializes large weight
 * buffers using TFLite's out-of-line `Buffer.Offset()`/`Size()` convention, which the
 * app's pinned `TensorFlowLiteC 2.17.0` doesn't resolve through either of its
 * model-loading APIs. `src/smart_scene_analyzer/export/tflite.py`'s `export_detector`
 * now inlines those buffers back into the classic format as a post-export step
 * (`_inline_offset_buffers`), which fixes it -- confirmed on-device, see
 * `docs/parity-evidence/round2-invoke-root-cause.md` for the full diagnosis.
 */
import { Image as RNImage } from 'react-native';

import { decodeImage, type RGBPixels, type Size } from '../inference/image';
import { toModelInput, type ChannelNorm } from '../inference/preprocess';
import { decodeDetections, type ModelInputBox } from '../inference/decode';
import { nms, NMS_IOU_THRESHOLD } from '../inference/nms';
import { loadDetectorModel } from '../inference/detector';
import { runDepthInference } from '../inference/depth';
import { fuse, contentRect, type DepthMap, type LetterboxTransform } from '../fusion';
import type { DetectionOverlayState, OverlayDetection } from '../components/DetectionOverlay';
import { timeAsync, timeSync } from '../dev/phaseTimers';

const SOURCE_LONG_SIDE = 640;
const DETECTOR_INPUT_SIZE = { width: 640, height: 640 };
const DETECTOR_NORM: ChannelNorm = { mean: [0, 0, 0], std: [1, 1, 1] };

export interface AnalyzeImageResult {
  /** SOURCE pixel size of the (already longSide-resized) decoded image -- see image.ts's
   * module header: this is NOT the original photo's resolution. */
  sourceSize: Size;
  /** The DETECTOR's own letterbox for this run -- the exact values DetectionOverlay's
   * toScreen call needs, never recomputed from scratch by a caller. */
  detectorLetterbox: { scale: number; padX: number; padY: number };
  overlayState: DetectionOverlayState;
}

function getImageSize(uri: string): Promise<Size> {
  return new Promise((resolve, reject) => {
    RNImage.getSize(uri, (width, height) => resolve({ width, height }), reject);
  });
}

/** Undoes the DETECTOR's letterbox to recover SOURCE pixels -- the space the real
 * fusion port (`../fusion`) requires its input boxes in (matches
 * `fusion.py`'s contract C7). This duplicates toScreen.ts's first step for a different
 * consumer (feeding the fusion layer, not placing a box on screen) --
 * DetectionOverlay.tsx never calls this; it calls modelInputPixelsToScreenPoints
 * directly on the ORIGINAL detector-space box, which is why `box` below is returned
 * unconverted alongside the SOURCE-space one used only for fusion's sampling. */
function detectorModelInputToSourcePixels(box: ModelInputBox, letterbox: { scale: number; padX: number; padY: number }) {
  return {
    x1: (box.x1 - letterbox.padX) / letterbox.scale,
    y1: (box.y1 - letterbox.padY) / letterbox.scale,
    x2: (box.x2 - letterbox.padX) / letterbox.scale,
    y2: (box.y2 - letterbox.padY) / letterbox.scale,
  };
}

async function decodeSource(uri: string): Promise<RGBPixels> {
  const originalSize = await getImageSize(uri);
  return decodeImage(uri, originalSize, SOURCE_LONG_SIDE);
}

/**
 * Runs the full detection + depth pipeline on one image, identified by `uri`. This is
 * the single function Photo mode and Live mode both call -- see this module's header
 * for exactly what's real vs. substituted on the detector side.
 */
export async function analyzeImageUri(uri: string): Promise<AnalyzeImageResult> {
  // 'preprocess' = decode (native resize + JPEG decode) + the DETECTOR's own letterbox.
  // Depth's own letterbox is timed separately as 'depthPreprocess', inside depth.ts --
  // it has a different target size (518 vs 640) and runs at a different point in this
  // function, so folding it into this number would hide which model's preprocessing
  // actually costs what. See docs/latency-report.md's "Preprocess" row for how the two
  // are reported together.
  const source = await timeAsync('preprocessDecode', () => decodeSource(uri));

  const detectorModelInput = timeSync('preprocessLetterbox', () =>
    toModelInput(source, DETECTOR_INPUT_SIZE, DETECTOR_NORM)
  );
  const detectorLetterbox = { scale: detectorModelInput.scale, padX: detectorModelInput.padX, padY: detectorModelInput.padY };

  // Cold load is self-timed and self-reported inside loadDetectorModel() (its own
  // `loadMs`, console-logged there) -- matches loadDepthModel()'s identical convention in
  // depth.ts. Cached after the first call, so this is near-instant on every later frame.
  const detectorState = await loadDetectorModel();
  if (detectorState.status !== 'ready') {
    throw new Error(
      `analyzeImageUri: detector model failed to load (status: ${detectorState.status})`
    );
  }

  const rawOutputs = timeSync('detectionInference', () =>
    // `Float32Array.buffer` is typed ArrayBufferLike (also covers SharedArrayBuffer);
    // toModelInput always constructs a fresh `new Float32Array(...)`, so this is always
    // a real ArrayBuffer at runtime.
    detectorState.model.runSync([detectorModelInput.data.buffer as ArrayBuffer])
  );
  const raw = new Float32Array(rawOutputs[0]!);
  // Read features/anchors back from the loaded model's own metadata (outputs[0].shape is
  // [1, features, anchors]) rather than a hardcoded constant -- matches detector.ts's own
  // "trust the loaded model, not the card" convention.
  const outputShape = detectorState.model.outputs[0]!.shape;
  const [, features, anchors] = outputShape as [number, number, number];
  const { decoded, kept } = timeSync('decodeNms', () => {
    const decoded = decodeDetections(raw, { anchors, features }, DETECTOR_INPUT_SIZE);
    const kept = nms(decoded, NMS_IOU_THRESHOLD);
    return { decoded, kept };
  });
  void decoded;

  // Real, on-device: depth.ts times its own letterbox ('depthPreprocess') and its own
  // execute() call ('depthInference') separately, internally -- see depth.ts's
  // runDepthInference for both.
  const depthResult = await runDepthInference(source);

  const detections: OverlayDetection[] = timeSync('fusion', () => {
    // The DEPTH model's own SOURCE -> DEPTH_INPUT letterbox transform -- NOT the
    // detector's (detectorLetterbox, above). `fuse` (the real fusion port) does the
    // SOURCE -> DEPTH_INPUT mapping itself, internally, given this transform -- it must
    // never be handed a box already converted into depth-input space.
    const depthTransform: LetterboxTransform = {
      scale: depthResult.scale,
      padX: depthResult.padX,
      padY: depthResult.padY,
      inW: source.size.width,
      inH: source.size.height,
      outW: depthResult.relativeDepth.width,
      outH: depthResult.relativeDepth.height,
    };
    const depthMap: DepthMap = {
      values: depthResult.relativeDepth.data,
      width: depthResult.relativeDepth.width,
      height: depthResult.relativeDepth.height,
      transform: depthTransform,
      valid: contentRect(depthTransform),
    };

    const fused = fuse(
      kept.map((d) => ({
        box: detectorModelInputToSourcePixels(d.box, detectorLetterbox),
        classId: d.classId,
      })),
      depthMap
    );

    return kept.map((d, i) => ({
      box: d.box,
      label: d.label,
      confidence: d.confidence,
      // OverlayDetection.relativeDepth is a required `number` (DetectionOverlay.tsx has
      // no depth_quality-aware rendering yet -- out of this port's scope, flagged as an
      // open item rather than silently redesigned here). `fuse` reports `null` when no
      // depth value was derivable (depthQuality 'unavailable'); 0 preserves this
      // pipeline's prior all-degenerate-boxes-draw-at-zero behavior for that case.
      relativeDepth: fused[i]!.relativeDepth ?? 0,
    }));
  });

  return {
    sourceSize: source.size,
    detectorLetterbox,
    overlayState: { status: 'ready', detections },
  };
}
