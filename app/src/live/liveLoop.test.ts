/**
 * Proves the skip-never-queue rule (module header, liveLoop.ts) without a real camera,
 * a real clip, or real wall-clock seconds -- this is the proof resources/prompts/
 * 06-live-mode.md calls for explicitly: "a slow `analyze` stub, advance the clock past
 * two intervals, assert one skip and no queue."
 *
 * NOTE: no test runner is wired into app/ yet (no jest/jest-expo in package.json -- see
 * decode.test.ts's header and the Lesson 05 step 6 Round 1 report; still true as of this
 * round). This file is written in standard jest syntax (describe/it/expect) so it runs
 * unmodified once a runner is added.
 *
 * It does NOT use `jest.useFakeTimers()` -- liveLoop.ts takes its `Clock` as a
 * constructor-style parameter specifically so a test never needs jest's global timer
 * mock at all. `FakeClock` below is a tiny hand-rolled double: `setInterval` just
 * remembers the callback, and `tick()` invokes it synchronously, standing in for "one
 * interval's worth of wall-clock time has now passed." That means this test's actual
 * assertions were verified for real, right now, by compiling this file (with the
 * surrounding describe/it/expect replaced by a plain runner) with `tsc` and executing
 * the result with `node` -- the same verification path every other *.test.ts in this
 * app/ has used since no runner exists (see preprocess.test.ts's header) -- not merely
 * reasoned about on paper.
 */
import { startLiveLoop, type Clock } from './liveLoop';

/** Remembers whatever callback liveLoop.ts's single `setInterval` call registers, and
 * lets the test fire it on command via `tick()`. There is only ever one live interval
 * per loop, so one remembered callback is enough -- this is not a general-purpose fake
 * timer library, just the minimum this one test needs. */
class FakeClock implements Clock {
  private callback: (() => void) | null = null;

  setInterval(callback: () => void): unknown {
    this.callback = callback;
    return 1;
  }

  clearInterval(): void {
    this.callback = null;
  }

  /** Simulates exactly one interval elapsing. */
  tick(): void {
    this.callback?.();
  }
}

/** Lets pending promise continuations (the .then chains inside liveLoop.ts's tick)
 * actually run before the next assertion -- a tick's capture/analyze calls are async,
 * so a synchronous tick() call alone does not let their callbacks execute yet. */
function flushMicrotasks(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe('startLiveLoop', () => {
  it('skips a capture due while the previous analyze is still in flight, and never queues it', async () => {
    const clock = new FakeClock();

    let analyzeCallCount = 0;
    let pendingResolve: (() => void) | null = null;
    // A SLOW analyze stub: it does not resolve until the test explicitly lets it.
    const analyze = (_uri: string) =>
      new Promise<string>((resolve) => {
        analyzeCallCount += 1;
        pendingResolve = () => resolve('result');
      });

    let captureCallCount = 0;
    const capture = async () => {
      captureCallCount += 1;
      return `uri-${captureCallCount}`;
    };

    const results: string[] = [];
    const handle = startLiveLoop<string>({
      intervalMs: 1000,
      capture,
      analyze,
      onResult: (result) => results.push(result),
      clock,
    });

    // Interval #1: nothing in flight yet -- this one must run.
    clock.tick();
    await flushMicrotasks();
    expect(handle.frameCount).toBe(1);
    expect(handle.skippedCount).toBe(0);
    expect(analyzeCallCount).toBe(1);
    expect(captureCallCount).toBe(1);

    // Interval #2: frame #1's analyze is still pending (pendingResolve hasn't been
    // called) -- this tick is due WHILE the last one is still in flight, so it must be
    // SKIPPED, not queued.
    clock.tick();
    await flushMicrotasks();
    expect(handle.frameCount).toBe(1); // unchanged -- no second frame started
    expect(handle.skippedCount).toBe(1); // the skip counted
    expect(analyzeCallCount).toBe(1); // analyze was NOT called again
    expect(captureCallCount).toBe(1); // capture was NOT called again either

    // A third interval, still in flight, proves this isn't a one-tick fluke -- past
    // TWO intervals, as the prompt specifies, with no queued work anywhere.
    clock.tick();
    await flushMicrotasks();
    expect(handle.frameCount).toBe(1);
    expect(handle.skippedCount).toBe(2);
    expect(analyzeCallCount).toBe(1);

    // Let frame #1 finish, then confirm the loop recovers: the NEXT interval runs a
    // fresh frame rather than replaying anything that was skipped (there is nothing
    // queued to replay).
    pendingResolve?.();
    await flushMicrotasks();
    expect(results).toEqual(['result']);

    clock.tick();
    await flushMicrotasks();
    expect(handle.frameCount).toBe(2);
    expect(handle.skippedCount).toBe(2); // unchanged by the successful tick
    expect(analyzeCallCount).toBe(2);

    handle.stop();
  });
});
