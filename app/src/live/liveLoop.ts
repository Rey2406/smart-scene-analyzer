/**
 * The throttle + in-flight-skip rule, extracted as pure logic so it can be unit-tested
 * without React, a camera, or real wall-clock seconds to wait through.
 *
 * Why a `Clock` parameter instead of calling `setInterval` directly: a real interval
 * fires on wall-clock time, which makes "skip the capture due while the last one is
 * still in flight" nearly impossible to test deterministically -- you'd be racing a
 * real timer against a real async stub. Taking an injectable `Clock` instead lets a
 * test command exact, synchronous time steps and assert exactly what fired on each one
 * (see liveLoop.test.ts). `realClock` is the only implementation the app itself uses.
 *
 * THE RULE THIS FILE ENFORCES, per resources/prompts/06-live-mode.md: if a capture is
 * due while the previous one is still being analyzed, SKIP it -- never queue it. A
 * queue grows without bound the moment analysis is slower than the interval (which, on
 * this project's JS-thread-blocking pipeline, is most of the time -- see the Round 3/4
 * report), and the overlay would fall further behind the real world every second. A
 * skipped frame costs nothing; a queued one costs a growing backlog.
 */

/** The minimal scheduler surface this module needs -- deliberately not the full
 * `setInterval`/`clearInterval` global signature, so a test's fake implementation has
 * nothing extra to stub. */
export interface Clock {
  setInterval(callback: () => void, ms: number): unknown;
  clearInterval(id: unknown): void;
}

export const realClock: Clock = {
  setInterval: (callback, ms) => setInterval(callback, ms),
  clearInterval: (id) => clearInterval(id as ReturnType<typeof setInterval>),
};

export interface LiveLoopOptions<TResult> {
  intervalMs: number;
  /** Produces one still's file URI. The capture SOURCE -- see capture/types.ts -- not
   * the pipeline itself. */
  capture: () => Promise<string>;
  /** The SAME function Photo mode calls (pipeline/analyzeImage.ts's analyzeImageUri).
   * Live mode must never substitute its own decode/letterbox/depth call here. */
  analyze: (uri: string) => Promise<TResult>;
  onResult: (result: TResult, uri: string) => void;
  onError?: (error: unknown) => void;
  /** Fired synchronously right after frameCount or skippedCount changes -- a React
   * caller uses this to mirror the counters into state, since mutating the handle's
   * fields in place does not itself trigger a re-render. */
  onCountsChanged?: (counts: { frameCount: number; skippedCount: number }) => void;
  clock?: Clock;
}

export interface LiveLoopHandle {
  /** Stops the interval. Capture/analyze already in flight still finishes and still
   * calls onResult/onError -- stopping only prevents new captures from starting. */
  stop(): void;
  /** How many captures actually ran (started analyze). Mutated in place on the same
   * object rather than replaced, so a caller can poll it straight into state each tick
   * without the handle itself being a moving target. */
  frameCount: number;
  /** How many capture ticks were skipped because the previous one was still in flight.
   * NEVER a queue length -- this is a counter of discarded ticks, nothing is retained. */
  skippedCount: number;
}

/**
 * Starts the throttle loop: every `intervalMs`, either starts a new capture+analyze (if
 * none is in flight) or increments `skippedCount` (if one is). Returns a handle whose
 * `frameCount`/`skippedCount` fields update in place -- call `stop()` to end it.
 */
export function startLiveLoop<TResult>(options: LiveLoopOptions<TResult>): LiveLoopHandle {
  const clock = options.clock ?? realClock;
  let inFlight = false;

  const handle: LiveLoopHandle = {
    frameCount: 0,
    skippedCount: 0,
    stop: () => {
      /* replaced below once timerId exists */
    },
  };

  const reportCounts = () =>
    options.onCountsChanged?.({ frameCount: handle.frameCount, skippedCount: handle.skippedCount });

  const tick = () => {
    if (inFlight) {
      // SKIP -- never queue. This is the whole rule; see module header.
      handle.skippedCount += 1;
      reportCounts();
      return;
    }
    inFlight = true;
    handle.frameCount += 1;
    reportCounts();

    (async () => {
      try {
        const uri = await options.capture();
        const result = await options.analyze(uri);
        options.onResult(result, uri);
      } catch (error) {
        options.onError?.(error);
      } finally {
        inFlight = false;
      }
    })();
  };

  const timerId = clock.setInterval(tick, options.intervalMs);
  handle.stop = () => clock.clearInterval(timerId);

  return handle;
}
