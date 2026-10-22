/**
 * Scene Understanding Layer -- the on-device PORT of
 * `src/smart_scene_analyzer/fusion.py` (Stage S6, the reference implementation).
 *
 * **This is a port, not a rewrite.** It is held to agreement with the Python reference
 * on the SAME fixtures (`tests/fixtures/fusion_cases.json`, see `index.test.ts`) and by
 * nothing else. When the two disagree, that disagreement is the finding -- fix this file,
 * or record why the reference is wrong. Never adjust the fixture to force agreement.
 *
 * **Units, stated once.** `relativeDepth` is relative inverse depth: larger is nearer,
 * the scale is arbitrary, and two values are comparable only when they came from the
 * same image. Nothing in this file may imply a unit -- no metric length word, no
 * physical distance anywhere in a name, a comment, or a UI-facing string.
 *
 * **Pure by construction.** Arrays and boxes in, plain objects out. No model, no I/O, no
 * React, no clock, no randomness.
 *
 * The reduction (per detection), mirroring fusion.py's module docstring exactly:
 *   1. Map the SOURCE-space box into DEPTH_INPUT space with the depth map's OWN
 *      letterbox transform (never the detector's -- that is a different transform).
 *   2. Intersect with `depthMap.valid` -- a letterbox padding pixel is never sampled.
 *   3. Inset by `insetFraction` of each side (default 0.125), floor-clamped: if that
 *      would collapse the box below 1px per side, fall back to the un-inset box instead
 *      (and report that fallback -- see `depthQuality`).
 *   4. Stride the region so at most `maxSamples` points are read.
 *   5. Drop non-finite samples.
 *   6. `relativeDepth` = the MEDIAN of the survivors (not the mean -- a background gap
 *      inside a box, the normal case for a non-convex object, moves the mean but not the
 *      median until background is the majority).
 *   7. `depthSpread` = IQR of the samples / the image's robust (2nd-98th percentile)
 *      range, clamped to [0, 1].
 *   8. `relativeDepthNormalized` = the median's position within that same robust range,
 *      clamped to [0, 1].
 *   9. `depthRank` = index after sorting descending by `relativeDepth` (ties broken by
 *      ascending `classId`); 0 is nearest. A detection with no value gets no rank.
 *
 * `depthQuality` precedence (fusion.py's `classify`): "unavailable" (no value at all) ->
 * "fallback_full_box" (the inset collapsed, so the un-inset box was sampled) ->
 * "mixed_region" (`depthSpread >= MIXED_REGION_SPREAD_THRESHOLD`, the box straddles a
 * real depth discontinuity) -> "ok".
 *
 * JUDGMENT CALL: the Python reference splits this logic across three files --
 * `fusion.py` plus `geometry/boxes.py` (`inset`, `intersect`) and `geometry/letterbox.py`
 * (`letterbox_forward`, `content_rect`). `app/src/` does not yet have geometry/boxes.ts
 * or geometry/letterbox.ts equivalents, and this port's brief is specifically
 * fusion.py's surface, so those small helpers are reproduced as private functions below
 * rather than spun into new modules speculatively. `contentRect` is exported anyway,
 * because the shared-fixture test (and any future caller building a `DepthMap` from a
 * raw transform) needs it for exactly the reason `test_fusion.py`'s loader does: `valid`
 * is deliberately not stored in the fixture, only derived from `transform`.
 */

import type { ModelInputBox } from '../inference/decode';

/** Generic xyxy, absolute-pixel box shape, reused from decode.ts's `ModelInputBox` --
 * same four named fields, any space. WHICH space a given box is in is determined by the
 * caller, exactly as the rest of this codebase already does for this field shape (e.g.
 * `ModelInputBox` itself is MODEL-INPUT space only by the convention of what produced
 * it). In this module's own contract, boxes passed to `fuse` / `fuseDetectionsWithDepth`
 * must be in SOURCE space -- matching fusion.py's contract C7 -- while `depthMap`
 * carries the SOURCE -> DEPTH_INPUT transform used internally to reach DEPTH_INPUT
 * pixels for sampling. */
export type BoxXYXY = ModelInputBox;

/** SOURCE -> one model-input space. Mirrors `smart_scene_analyzer.types.LetterboxTransform`.
 * Forward: `x_model = x_source * scale + padX`. */
export interface LetterboxTransform {
  scale: number;
  padX: number;
  padY: number;
  inW: number;
  inH: number;
  outW: number;
  outH: number;
}

/** C8 -- the dense depth output, in DEPTH_INPUT space. `values` is row-major,
 * length === width * height. Units: none -- relative inverse depth, larger is nearer,
 * comparable only within this one image. */
export interface DepthMap {
  values: Float32Array;
  width: number;
  height: number;
  transform: LetterboxTransform;
  /** The un-padded content rectangle, xyxy in DEPTH_INPUT pixels. Padding is never
   * sampled. Usually `contentRect(transform)`. */
  valid: BoxXYXY;
}

/** The minimal shape `fuse` needs from a detection: a SOURCE-space box and the class id
 * used to break rank ties. Callers pass their own richer detection type (label,
 * confidence, ...) and get the exact same object back on `FusedDetection.detection`. */
export interface FusionDetectionInput {
  box: BoxXYXY;
  classId: number;
}

export type DepthQuality = 'ok' | 'unavailable' | 'fallback_full_box' | 'mixed_region';

/** C9 -- one detected object with its depth score attached. See this module's header
 * for exactly what each field means and how `depthQuality` is classified. */
export interface FusedDetection<D extends FusionDetectionInput = FusionDetectionInput> {
  detection: D;
  relativeDepth: number | null;
  relativeDepthNormalized: number | null;
  depthRank: number | null;
  depthSpread: number | null;
  depthSampleCount: number;
  depthQuality: DepthQuality;
}

/** The per-box reduction result. Unit-less; larger `median` is nearer. */
export interface BoxDepthStatistic {
  median: number | null;
  spread: number | null;
  sampleCount: number;
  fellBackToFullBox: boolean;
}

/** `depthSpread` below this treats the image's dynamic range as degenerate: everything
 * reads as mid-scale rather than dividing by ~0. Matches fusion.py's `_ROBUST_RANGE_EPS`. */
const ROBUST_RANGE_EPS = 1e-9;

/** `depthSpread` at or above this is reported as `"mixed_region"`. Matches fusion.py's
 * `MIXED_REGION_SPREAD_THRESHOLD` -- a judgment call, named so it is one place to
 * revisit, not a number buried in a conditional. */
export const MIXED_REGION_SPREAD_THRESHOLD = 0.3;

const DEFAULT_INSET_FRACTION = 0.125;
const DEFAULT_MAX_SAMPLES = 1024;

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

// ---------------------------------------------------------------- geometry (private port)
// Mirrors smart_scene_analyzer/geometry/boxes.py (`inset`, `intersect`) and
// geometry/letterbox.py (`letterbox_forward`, `content_rect`). See the module header's
// JUDGMENT CALL note for why these live here rather than in a separate geometry module.

/** xyxy absolute px, SOURCE -> model-input. Unclipped. Mirrors `letterbox_forward`. */
function letterboxForward(box: BoxXYXY, t: LetterboxTransform): BoxXYXY {
  return {
    x1: box.x1 * t.scale + t.padX,
    y1: box.y1 * t.scale + t.padY,
    x2: box.x2 * t.scale + t.padX,
    y2: box.y2 * t.scale + t.padY,
  };
}

/** The un-padded region of a model-input canvas, xyxy in model-input px. Everything
 * outside it is padding and must not be sampled. Mirrors `content_rect`. Exported: the
 * shared-fixture test derives `DepthMap.valid` from `transform` with this, exactly as
 * `test_fusion.py`'s loader does on the Python side. */
export function contentRect(t: LetterboxTransform): BoxXYXY {
  const newW = t.inW * t.scale;
  const newH = t.inH * t.scale;
  return { x1: t.padX, y1: t.padY, x2: t.padX + newW, y2: t.padY + newH };
}

/** `null` when the boxes do not overlap -- not a zero-area box, which would be
 * indistinguishable from a legitimate degenerate one. Mirrors `intersect`. */
function intersectBoxes(a: BoxXYXY, b: BoxXYXY): BoxXYXY | null {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  if (ix2 <= ix1 || iy2 <= iy1) return null;
  return { x1: ix1, y1: iy1, x2: ix2, y2: iy2 };
}

/** Shrink by `fraction` of each side, keeping the centre. Floor-clamped: if that would
 * take either side below `minSidePx`, returns `box` UNCHANGED -- `reduceBox` detects
 * this exact case (by value, not by reference) to set `fellBackToFullBox`. Mirrors
 * `inset`. */
function insetBox(box: BoxXYXY, fraction: number, minSidePx: number): BoxXYXY {
  const w = box.x2 - box.x1;
  const h = box.y2 - box.y1;
  const dx = w * fraction;
  const dy = h * fraction;
  const newW = w - 2 * dx;
  const newH = h - 2 * dy;
  if (newW < minSidePx || newH < minSidePx) return box;
  return { x1: box.x1 + dx, y1: box.y1 + dy, x2: box.x2 - dx, y2: box.y2 - dy };
}

// --------------------------------------------------------------------- numeric reduction

/** Percentile with NumPy's default ('linear') interpolation, over an ALREADY SORTED
 * ascending array. Used for the robust range (2nd/98th) and the IQR (25th/75th) --
 * never for the median; see `computeMedian` below for why that stays separate. */
function percentileSorted(sortedAscending: number[], p: number): number {
  const n = sortedAscending.length;
  if (n === 1) return sortedAscending[0]!;
  const rank = (p / 100) * (n - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) return sortedAscending[lower]!;
  const weight = rank - lower;
  return sortedAscending[lower]! + weight * (sortedAscending[upper]! - sortedAscending[lower]!);
}

/** The MEDIAN of an already-sorted ascending array: the middle value for an odd count,
 * the AVERAGE of the two middle values for an even count -- matching NumPy's
 * `np.median` exactly. Deliberately its own function, not routed through
 * `percentileSorted(_, 50)` (even though the two are mathematically identical for the
 * linear-interpolation method): Round 3 of this port's brief sabotages ONLY this
 * function (swap the average for the lower-middle value) to prove the fixture suite
 * actually watches the median definition, and that sabotage must not also touch
 * `depthSpread`/`robustRange`'s percentiles. */
function computeMedian(sortedAscending: number[]): number {
  const n = sortedAscending.length;
  const mid = Math.floor(n / 2);
  if (n % 2 === 1) return sortedAscending[mid]!;
  return (sortedAscending[mid - 1]! + sortedAscending[mid]!) / 2;
}

/** The 2nd and 98th percentiles of the map inside `valid`. Unit-less, meaningful only
 * within this one image. Mirrors `robust_range`. Recomputed on every call that needs
 * it (including once per box, inside `reduceBox`) -- matching the Python reference's own
 * (non-optimized) structure exactly, not "fixed" here, because parity is the point. */
export function robustRange(depthMap: DepthMap): [number, number] {
  const { x1, y1, x2, y2 } = depthMap.valid;
  const w = depthMap.width;
  const h = depthMap.height;
  const px1 = Math.max(0, Math.floor(x1));
  const py1 = Math.max(0, Math.floor(y1));
  const px2 = Math.min(w, Math.ceil(x2));
  const py2 = Math.min(h, Math.ceil(y2));

  const finite: number[] = [];
  for (let y = py1; y < py2; y++) {
    for (let x = px1; x < px2; x++) {
      const v = depthMap.values[y * w + x]!;
      if (Number.isFinite(v)) finite.push(v);
    }
  }
  if (finite.length === 0) return [0, 0];
  finite.sort((a, b) => a - b);
  return [percentileSorted(finite, 2), percentileSorted(finite, 98)];
}

/** The per-box reduction (module header, steps 1-7), for one box. Mirrors `reduce_box`. */
export function reduceBox(
  depthMap: DepthMap,
  boxSource: BoxXYXY,
  insetFraction: number,
  maxSamples: number
): BoxDepthStatistic {
  const boxDepth = letterboxForward(boxSource, depthMap.transform);
  const region = intersectBoxes(boxDepth, depthMap.valid);
  if (region === null) {
    // Box lies entirely in the padding band.
    return { median: null, spread: null, sampleCount: 0, fellBackToFullBox: false };
  }

  const sampled = insetBox(region, insetFraction, 1.0);
  const fellBack =
    insetFraction > 0.0 &&
    sampled.x1 === region.x1 &&
    sampled.y1 === region.y1 &&
    sampled.x2 === region.x2 &&
    sampled.y2 === region.y2;

  const w = depthMap.width;
  const h = depthMap.height;
  const px1 = Math.max(0, Math.floor(sampled.x1));
  const py1 = Math.max(0, Math.floor(sampled.y1));
  const px2 = Math.min(w, Math.ceil(sampled.x2));
  const py2 = Math.min(h, Math.ceil(sampled.y2));
  if (px2 <= px1 || py2 <= py1) {
    return { median: null, spread: null, sampleCount: 0, fellBackToFullBox: false };
  }

  const regionW = px2 - px1;
  const regionH = py2 - py1;
  const area = regionW * regionH;
  const stride = area > maxSamples ? Math.max(1, Math.ceil(Math.sqrt(area / maxSamples))) : 1;

  const finite: number[] = [];
  for (let y = py1; y < py2; y += stride) {
    for (let x = px1; x < px2; x += stride) {
      const v = depthMap.values[y * w + x]!;
      if (Number.isFinite(v)) finite.push(v);
    }
  }
  if (finite.length === 0) {
    return { median: null, spread: null, sampleCount: 0, fellBackToFullBox: false };
  }

  const sortedFinite = [...finite].sort((a, b) => a - b);
  const median = computeMedian(sortedFinite);

  const [lo, hi] = robustRange(depthMap);
  const span = hi - lo;
  let spread: number;
  if (span < ROBUST_RANGE_EPS) {
    spread = 0.0;
  } else {
    const q25 = percentileSorted(sortedFinite, 25);
    const q75 = percentileSorted(sortedFinite, 75);
    spread = clamp((q75 - q25) / span, 0.0, 1.0);
  }

  return { median, spread, sampleCount: finite.length, fellBackToFullBox: fellBack };
}

// ------------------------------------------------------------------------------- fuse()

/**
 * Attach a relative depth score to each detection. Boxes are unchanged and stay in
 * SOURCE space. Ranks are assigned across the returned array, `0` = nearest, among
 * detections that have a value -- a detection with no value gets no rank, not a gap in
 * the sequence (N12: `depthMap === null` degrades every detection the same way).
 * Mirrors `fuse`.
 *
 * Every degenerate case has a defined, non-NaN result -- `null`, never `NaN`, reaches
 * the caller.
 */
export function fuse<D extends FusionDetectionInput>(
  detections: readonly D[],
  depthMap: DepthMap | null,
  insetFraction: number = DEFAULT_INSET_FRACTION,
  maxSamples: number = DEFAULT_MAX_SAMPLES
): FusedDetection<D>[] {
  if (detections.length === 0) return [];

  if (depthMap === null) {
    return detections.map((d) => ({
      detection: d,
      relativeDepth: null,
      relativeDepthNormalized: null,
      depthRank: null,
      depthSpread: null,
      depthSampleCount: 0,
      depthQuality: 'unavailable',
    }));
  }

  const stats = detections.map((d) => reduceBox(depthMap, d.box, insetFraction, maxSamples));
  const [lo, hi] = robustRange(depthMap);
  const span = hi - lo;

  const normalize = (median: number | null): number | null => {
    if (median === null) return null;
    if (span < ROBUST_RANGE_EPS) return 0.5;
    return clamp((median - lo) / span, 0.0, 1.0);
  };

  const classify = (stat: BoxDepthStatistic): DepthQuality => {
    if (stat.median === null) return 'unavailable';
    if (stat.fellBackToFullBox) return 'fallback_full_box';
    if (stat.spread !== null && stat.spread >= MIXED_REGION_SPREAD_THRESHOLD) return 'mixed_region';
    return 'ok';
  };

  const validIndices = stats
    .map((_, i) => i)
    .filter((i) => stats[i]!.median !== null)
    .sort((a, b) => {
      const medianDiff = stats[b]!.median! - stats[a]!.median!; // descending median
      if (medianDiff !== 0) return medianDiff;
      return detections[a]!.classId - detections[b]!.classId; // ascending classId tiebreak
    });

  const ranks: (number | null)[] = new Array(detections.length).fill(null);
  validIndices.forEach((i, rank) => {
    ranks[i] = rank;
  });

  return detections.map((d, i) => ({
    detection: d,
    relativeDepth: stats[i]!.median,
    relativeDepthNormalized: normalize(stats[i]!.median),
    depthRank: ranks[i]!,
    depthSpread: stats[i]!.spread,
    depthSampleCount: stats[i]!.sampleCount,
    depthQuality: classify(stats[i]!),
  }));
}

/**
 * Convenience wrapper over `fuse` for callers that only have a raw depth array at
 * SOURCE resolution (an `HxW` buffer), not a `DepthMap` with its own letterbox
 * transform. Mirrors `fuse_detections_with_depth`: wraps the raw array in an IDENTITY
 * transform (scale=1, no padding) rather than silently treating SOURCE and DEPTH_INPUT
 * as the same space everywhere else in this codebase does not.
 *
 * NOTE: this is the real ported surface, not the prompt's illustrative
 * `fuseDetectionsWithDepth(detections, depthMap)` signature -- the actual Python
 * reference exports `fuse` (the primary, `DepthMap`-based function) and this wrapper
 * (for a raw array) as two distinct functions, and this port reproduces that surface
 * rather than the simplified one.
 */
export function fuseDetectionsWithDepth<D extends FusionDetectionInput>(
  detections: readonly D[],
  depthValues: Float32Array | null,
  imageWidth: number,
  imageHeight: number
): FusedDetection<D>[] {
  if (depthValues === null) return fuse(detections, null);
  if (depthValues.length !== imageWidth * imageHeight) {
    throw new Error(
      `fuseDetectionsWithDepth: depthValues length ${depthValues.length} does not match ` +
        `the stated frame size ${imageWidth}x${imageHeight} -- pass the array this frame ` +
        'actually produced, not a cached one from a different resolution.'
    );
  }
  const identity: LetterboxTransform = {
    scale: 1.0,
    padX: 0.0,
    padY: 0.0,
    inW: imageWidth,
    inH: imageHeight,
    outW: imageWidth,
    outH: imageHeight,
  };
  const wrapped: DepthMap = {
    values: depthValues,
    width: imageWidth,
    height: imageHeight,
    transform: identity,
    valid: { x1: 0, y1: 0, x2: imageWidth, y2: imageHeight },
  };
  return fuse(detections, wrapped);
}

/** Nearest (`depthRank` 0) first. A detection with no depth value sorts last -- it was
 * never ranked, not ranked worst. Mirrors `sort_near_to_far`. */
export function sortNearToFar<D extends FusionDetectionInput>(
  fused: readonly FusedDetection<D>[]
): FusedDetection<D>[] {
  return [...fused].sort((a, b) => {
    const aUnranked = a.depthRank === null ? 1 : 0;
    const bUnranked = b.depthRank === null ? 1 : 0;
    if (aUnranked !== bUnranked) return aUnranked - bUnranked;
    return (a.depthRank ?? 0) - (b.depthRank ?? 0);
  });
}
