/**
 * MODEL-INPUT pixels -> SCREEN points. The third and last coordinate-space boundary in
 * this pipeline (see app/CLAUDE.md's architecture diagram): SOURCE image pixels ->
 * (letterbox, preprocess.ts) -> MODEL-INPUT pixels -> (decode.ts, nms.ts, still
 * MODEL-INPUT) -> (this file) -> SCREEN points, what gets drawn.
 *
 * Lesson 04's fusion bug mixed spaces that all held numbers in the low hundreds -- a
 * wrong transform here would do the same thing, silently, because a box in the wrong
 * space is still a plausible-looking box. The fix is the same one fusion.ts used:
 * exactly one named function per transform, both spaces in the signature, no inline
 * arithmetic anywhere else.
 *
 * TWO letterboxes are composed here, not one:
 *   1. The MODEL's letterbox (`letterbox` param) -- produced by preprocess.ts's
 *      toModelInput when the SOURCE image was packed into the model's square canvas.
 *      Undoing it is step one, below, and it must come FIRST: the padding is in
 *      MODEL-INPUT pixels, and dividing by scale before subtracting the pad would scale
 *      the pad along with everything else, which is a different (wrong) number.
 *   2. The VIEW's own letterbox -- the on-screen `<Image>` almost never shares the
 *      source photo's aspect ratio (a 730x530 photo in a flexible-width card, say), so
 *      displaying it introduces a SECOND scale-and-offset, independent of the model's.
 *
 * JUDGMENT CALL: the view fits the image WITHIN its bounds (letterboxed, like
 * `resizeMode="contain"`), never crops it. A detection box can sit anywhere in the
 * source image, including near an edge a crop would cut off -- cropping the display
 * would silently discard a region a box was drawn against. This matches the `<Image
 * resizeMode="contain">` DetectionOverlay.tsx (Round 3) actually renders, so the boxes
 * drawn here land on pixels the user can actually see.
 */

import type { ModelInputBox } from '../inference/decode';

/** A box in MODEL-INPUT pixel space -- the detector's own padded square canvas, xyxy,
 * absolute pixels. Same shape as decode.ts's `ModelInputBox`; aliased under this more
 * explicit name (space is readable at the call site) rather than redefined, so the two
 * interfaces -- decode.ts's and this file's -- cannot drift apart. */
export type BoxXYXYModelInputPixels = ModelInputBox;

/** A box in SCREEN space: density-independent points (the same unit `onLayout` and
 * React Native's `style` dimensions use), origin at the top-left of the displaying
 * view. NOT device pixels -- multiply by `PixelRatio.get()` only if a consumer
 * specifically needs device pixels (none does today). */
export interface BoxXYXYScreenPoints {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Clamps `value` into `[min, max]`. A box partly outside the frame is normal (a
 * detection near the image edge, or the view's own aspect-ratio bars) -- clamping each
 * coordinate independently keeps the box on-screen without distorting the coordinates
 * that were already in range. */
function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Converts one box from MODEL-INPUT pixel space to SCREEN point space.
 *
 * Two steps, in order:
 *   1. Undo the model's letterbox: subtract `letterbox.padX`/`padY`, then divide by
 *      `letterbox.scale` -- recovers SOURCE-image pixels. `letterbox` must be the exact
 *      values `toModelInput` (preprocess.ts) returned for THIS image; recomputing them
 *      here would be a second copy of that arithmetic, and a second copy is how the two
 *      drift apart after the next edit to either one.
 *   2. Map SOURCE pixels into the view: the view fits the source image WITHIN its own
 *      bounds (never crops -- see module header), so this is its own scale + offset,
 *      independent of step 1's.
 *
 * Pure: no React, no hooks, no I/O, no mutation of any argument.
 */
export function modelInputPixelsToScreenPoints(
  box: BoxXYXYModelInputPixels,
  letterbox: { scale: number; padX: number; padY: number },
  sourceImageSize: { width: number; height: number },
  screenViewSize: { width: number; height: number }
): BoxXYXYScreenPoints {
  // Step 1 -- undo the MODEL letterbox: subtract padding FIRST, then divide by scale.
  const sourceX1 = (box.x1 - letterbox.padX) / letterbox.scale;
  const sourceY1 = (box.y1 - letterbox.padY) / letterbox.scale;
  const sourceX2 = (box.x2 - letterbox.padX) / letterbox.scale;
  const sourceY2 = (box.y2 - letterbox.padY) / letterbox.scale;

  // Step 2 -- map SOURCE pixels into the VIEW, which fits the image within its own
  // bounds (letterboxed, never cropped -- see module header). Isotropic scale, same
  // "smaller ratio wins" rule toModelInput uses, so the image is never stretched.
  const imageAspect = sourceImageSize.width / sourceImageSize.height;
  const viewAspect = screenViewSize.width / screenViewSize.height;

  let displayScale: number;
  let viewOffsetX: number;
  let viewOffsetY: number;

  if (imageAspect > viewAspect) {
    // Image is relatively wider than the view -- fits to the view's width, with bars
    // above and below (the view's OWN vertical letterbox).
    displayScale = screenViewSize.width / sourceImageSize.width;
    const displayHeight = sourceImageSize.height * displayScale;
    viewOffsetX = 0;
    viewOffsetY = (screenViewSize.height - displayHeight) / 2;
  } else {
    // Image is relatively taller than (or equal to) the view -- fits to the view's
    // height, with bars left and right.
    displayScale = screenViewSize.height / sourceImageSize.height;
    const displayWidth = sourceImageSize.width * displayScale;
    viewOffsetX = (screenViewSize.width - displayWidth) / 2;
    viewOffsetY = 0;
  }

  const screenX1 = viewOffsetX + sourceX1 * displayScale;
  const screenY1 = viewOffsetY + sourceY1 * displayScale;
  const screenX2 = viewOffsetX + sourceX2 * displayScale;
  const screenY2 = viewOffsetY + sourceY2 * displayScale;

  // Clamp to the view's own bounds -- a box partly outside the frame (near a source
  // edge, or larger than the source after a bad detection) is normal; it must not be
  // drawn past where the view actually is.
  return {
    x1: clamp(screenX1, 0, screenViewSize.width),
    y1: clamp(screenY1, 0, screenViewSize.height),
    x2: clamp(screenX2, 0, screenViewSize.width),
    y2: clamp(screenY2, 0, screenViewSize.height),
  };
}
