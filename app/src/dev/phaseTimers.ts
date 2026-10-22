/**
 * Lesson 05 step 11 — per-phase latency instrumentation, DEV-ONLY.
 *
 * Every export in this module is a no-op in a release build. `__DEV__` is a React
 * Native global, true only in a debug bundle -- Metro's minifier/dead-code-elimination
 * strips the `if (__DEV__)` branches below entirely in a release build, so this module
 * costs nothing in the product this project ships. It exists ONLY to answer
 * docs/latency-report.md's Round 2: "time each phase separately."
 *
 * Deliberately NOT a generic perf library -- a fixed, named set of phases
 * (see PhaseName) matching docs/latency-report.md's own table rows, so a sample
 * recorded here maps 1:1 onto a report row with no renaming step in between.
 *
 * `relativeDepth`/`depth_score` naming is irrelevant here -- this module measures
 * milliseconds, not depth values, so app/CLAUDE.md's units_guard rule does not apply to
 * it, and nothing here reads or forwards a depth number.
 */

export type PhaseName =
  | 'preprocessDecode' // native resize (expo-image-manipulator) + JPEG decode -> SOURCE RGB pixels
  | 'preprocessLetterbox' // detector's own letterbox + normalize (SOURCE -> DET_INPUT). Timed separately from preprocessDecode because the two run on different threads (native image manipulator vs. JS) and Round 2's first pass showed wildly different variance between them -- combining them hid which one dominates p95, exactly what the prompt warns a single aggregate number does
  | 'depthPreprocess' // depth model's own letterbox + normalize, timed separately because it has a different input size (518 vs 640) and a different cost
  | 'detectionInference' // real on-device Invoke() timing -- see decode.ts's header; UNUSED while that blocker stands, kept so the row exists if it is ever unblocked
  | 'depthInference' // the real on-device execute() call only, no preprocessing folded in
  | 'decodeNms' // decodeDetections() + nms()
  | 'fusion' // un-letterbox + depth-rect mapping + medianInRegion, per kept detection
  | 'render'; // setResult commit -> next painted frame, via double requestAnimationFrame

const samples: Record<PhaseName, number[]> = {
  preprocessDecode: [],
  preprocessLetterbox: [],
  depthPreprocess: [],
  detectionInference: [],
  depthInference: [],
  decodeNms: [],
  fusion: [],
  render: [],
};

/** Records one sample for `phase`, in milliseconds. No-op outside __DEV__ -- callers
 * still call this unconditionally; the branch, not the call site, is what disappears. */
export function recordPhase(phase: PhaseName, ms: number): void {
  if (!__DEV__) return;
  samples[phase].push(ms);
}

/** Times a synchronous function and records it under `phase`, returning the function's
 * own result unchanged. In a release build this still RUNS fn (correctness must not
 * depend on __DEV__) but skips the Date.now() calls and the push. */
export function timeSync<T>(phase: PhaseName, fn: () => T): T {
  if (!__DEV__) return fn();
  const started = Date.now();
  const result = fn();
  recordPhase(phase, Date.now() - started);
  return result;
}

/** Times an async function and records it under `phase`. Same no-op-outside-__DEV__
 * shape as timeSync, for the phases that await a native call (depth's execute(), the
 * image decode). */
export async function timeAsync<T>(phase: PhaseName, fn: () => Promise<T>): Promise<T> {
  if (!__DEV__) return fn();
  const started = Date.now();
  const result = await fn();
  recordPhase(phase, Date.now() - started);
  return result;
}

export function getSamples(phase: PhaseName): readonly number[] {
  return samples[phase];
}

export function resetSamples(): void {
  for (const phase of Object.keys(samples) as PhaseName[]) {
    samples[phase] = [];
  }
}

/** p50/p95 over `values`, nearest-rank method (no interpolation -- simplest to hand
 * verify, and the warm-run sample sizes here (n=20-30) are too small for interpolation
 * to matter). Returns NaN for an empty array rather than throwing, so a phase that
 * genuinely has zero samples (e.g. detectionInference, while blocked) is visible as
 * NaN -- not silently rendered as 0, which would read as "measured, and fast." */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, rank)]!;
}

export function summarize(phase: PhaseName): { n: number; p50: number; p95: number; mean: number } {
  const values = samples[phase];
  const mean = values.length === 0 ? NaN : values.reduce((a, b) => a + b, 0) / values.length;
  return { n: values.length, p50: percentile(values, 50), p95: percentile(values, 95), mean };
}

export function summarizeAll(): Record<PhaseName, { n: number; p50: number; p95: number; mean: number }> {
  const out = {} as Record<PhaseName, { n: number; p50: number; p95: number; mean: number }>;
  for (const phase of Object.keys(samples) as PhaseName[]) {
    out[phase] = summarize(phase);
  }
  return out;
}
