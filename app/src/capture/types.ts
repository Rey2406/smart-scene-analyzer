/**
 * The capture source boundary Live mode is built on. One tiny interface -- produce a
 * still, as a file URI -- so the throttle loop (live/liveLoop.ts) and the Live tab never
 * know or care whether the still came from a camera, a recorded clip, or a bundled test
 * photo. Swapping the source is swapping which object implements this; nothing else
 * changes, including the fake-timer unit test, which uses a fourth implementation that
 * exists only in the test.
 *
 * Deliberately NOT responsible for decoding pixels -- that is analyzeImage.ts's job,
 * once, for whichever source produced the URI. A capture source that decoded pixels
 * itself would be a second entry point into the pipeline, which is exactly what this
 * project's "one pipeline function" rule forbids (see pipeline/analyzeImage.ts's header).
 */
export interface StillCaptureSource {
  /** Produces one still image as a local file URI. Rejects if no still is currently
   * available (e.g. no clip picked yet) -- the caller decides what that means on screen. */
  captureStill(): Promise<string>;
}
