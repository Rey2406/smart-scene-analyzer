/**
 * Pure RGB-pixels -> model-input-tensor conversion: letterbox (never stretch) and
 * normalize. No dependency on any ML runtime -- this is plain arithmetic over typed
 * arrays, which is what makes it testable without a model, a device, or a simulator.
 *
 * Layout: [1, 3, H, W] NCHW float32, contiguous (channel-major -- three whole planes,
 * not interleaved RGBRGB...). This is NOT guessed. It is confirmed two ways:
 *   1. docs/model-card-v1.md's VERIFIED export table states depth.pte's input tensor
 *      is `[1, 3, 518, 518]` -- the traced PyTorch module's own declared dimension
 *      order, preserved end-to-end by ExecuTorch's `to_edge_transform_and_lower`.
 *   2. depth.ts reads the SAME fact back out of the *loaded* model at runtime
 *      (`model.schema['forward'].inputs[0]`, typed as `TensorSpec<ConcreteDim>` in
 *      react-native-executorch's own node_modules/react-native-executorch/src/core/
 *      schema.ts) and warns if the two ever disagree -- see depth.ts's
 *      EXPECTED_INPUT_SHAPE check. Trust the loaded model over this comment if they
 *      ever do.
 * NHWC was considered and rejected: nothing in react-native-executorch's own vision
 * task pipelines (src/extensions/cv/utils/imagePreprocessor.ts, which builds the same
 * `[1, 3, H, W]` layout for its bundled detection/segmentation tasks) or in Depth
 * Anything V2's ViT-S/14 patch-embedding conv (channel-first in PyTorch, the framework
 * the export traced) implies channel-last.
 *
 * LETTERBOX, not stretch: the source image is scaled isotropically (same factor on
 * both axes) to fit inside `target`, and centered -- the slack axis gets padding, the
 * image never gets distorted into the model's aspect ratio. `scale`, `padX`, and
 * `padY` are returned rather than left for a caller to recompute later: inverting a
 * depth-map region back to source-image pixels needs the EXACT values used here, and a
 * second computation elsewhere -- even one that looks identical -- is how the two
 * drift apart after the next edit to either copy.
 *
 * Padding fill is a JUDGMENT CALL: the model card does not state a pad color. Letterbox
 * bars are filled as pixel value 0 (black), run through the same per-channel
 * `(pixel/255 - mean) / std` formula as every real pixel, rather than left as raw
 * zeros in the normalized tensor. This matches Depth Anything V2's own Hugging Face
 * preprocessor (`DPTImageProcessor`), which pads before normalizing, not after. Nothing
 * downstream reads a depth value from the bar region, so this choice only affects
 * pixels no one uses -- but it is a choice, not a fact, and is recorded here as one.
 */
import type { RGBPixels } from './image';

/** Per-channel normalization constants, in R, G, B order. From the model card -- never
 * guessed and never hardcoded a second time inside this function. */
export interface ChannelNorm {
  mean: readonly [number, number, number];
  std: readonly [number, number, number];
}

/** The model's declared input tensor, flattened NCHW float32, plus the letterbox
 * transform that produced it -- the only facts needed to invert a result back to
 * source-image pixels later. */
export interface ModelInputTensor {
  /** Flat NCHW float32 buffer, length 1 * 3 * target.height * target.width. */
  data: Float32Array;
  /** Isotropic scale applied to the source image: sourcePx * scale -> pre-pad model-input px. */
  scale: number;
  /** Horizontal letterbox padding, in model-input pixels, applied before the scaled image. */
  padX: number;
  /** Vertical letterbox padding, in model-input pixels, applied before the scaled image. */
  padY: number;
}

/**
 * Letterboxes `pixels` into a `target.width` x `target.height` canvas (same scale on
 * both axes, centered, padded -- never stretched to fit), then applies
 * `(pixel/255 - mean) / std` per channel and returns the flat NCHW float32 tensor.
 */
export function toModelInput(
  pixels: RGBPixels,
  target: { width: number; height: number },
  norm: ChannelNorm
): ModelInputTensor {
  const { width: srcW, height: srcH } = pixels.size;
  const { width: dstW, height: dstH } = target;

  if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) {
    throw new Error(
      `toModelInput: invalid size (source ${srcW}x${srcH}, target ${dstW}x${dstH})`
    );
  }

  // Isotropic scale -- the SAME factor on both axes, so the aspect ratio survives.
  // Using the smaller of the two per-axis ratios guarantees the scaled image fits
  // inside target on both axes (the larger ratio would overflow one of them).
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const scaledW = Math.round(srcW * scale);
  const scaledH = Math.round(srcH * scale);
  // Center the scaled image: half the leftover space goes before it as padding, the
  // (possibly one-pixel-larger, when the leftover is odd) remainder goes after.
  const padX = Math.floor((dstW - scaledW) / 2);
  const padY = Math.floor((dstH - scaledH) / 2);

  const { mean, std } = norm;
  // Letterbox bars: pixel value 0 (black), normalized by the same formula as every
  // other pixel -- see module header's "Padding fill" note.
  const padNormalized: [number, number, number] = [
    (0 / 255 - mean[0]) / std[0],
    (0 / 255 - mean[1]) / std[1],
    (0 / 255 - mean[2]) / std[2],
  ];

  const data = new Float32Array(3 * dstH * dstW);
  const planeSize = dstH * dstW;

  for (let ty = 0; ty < dstH; ty++) {
    const inBoundsY = ty >= padY && ty < padY + scaledH;
    for (let tx = 0; tx < dstW; tx++) {
      const inBoundsX = tx >= padX && tx < padX + scaledW;
      const planeOffset = ty * dstW + tx;

      if (!inBoundsY || !inBoundsX) {
        data[0 * planeSize + planeOffset] = padNormalized[0];
        data[1 * planeSize + planeOffset] = padNormalized[1];
        data[2 * planeSize + planeOffset] = padNormalized[2];
        continue;
      }

      // Nearest-neighbor: map the model-input pixel back to the source pixel it was
      // scaled from. Deterministic and simple to hand-verify, which the test below
      // depends on. Clamped so float rounding at the far edge never reads past the
      // source buffer.
      const sx = Math.min(srcW - 1, Math.floor((tx - padX) / scale));
      const sy = Math.min(srcH - 1, Math.floor((ty - padY) / scale));
      const srcOffset = (sy * srcW + sx) * 3;

      const r = pixels.data[srcOffset]!;
      const g = pixels.data[srcOffset + 1]!;
      const b = pixels.data[srcOffset + 2]!;

      data[0 * planeSize + planeOffset] = (r / 255 - mean[0]) / std[0];
      data[1 * planeSize + planeOffset] = (g / 255 - mean[1]) / std[1];
      data[2 * planeSize + planeOffset] = (b / 255 - mean[2]) / std[2];
    }
  }

  return { data, scale, padX, padY };
}
