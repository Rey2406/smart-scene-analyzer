# On-device latency report

> A latency figure without its measurement conditions is not a measurement. This template
> puts the conditions above the numbers so they cannot be quoted apart from them.

## Prediction, written before measuring

> Fill this in **first** and do not edit it afterwards. The gap between prediction and
> measurement is the most useful line in this report.

| Phase | Predicted p50 |
|---|---|
| Cold load — detection | 400 ms |
| Cold load — depth | 1500 ms |
| Preprocess | 60 ms |
| Detection inference | 15 ms |
| Depth inference | 80 ms |
| Decode + NMS | 5 ms |
| Fusion | 3 ms |
| Render | 50 ms |
| Peak memory, both models loaded | 350 MB |

> **Basis for these guesses**, recorded so the gap is checkable rather than vibes:
> `docs/architecture.md` §6.1 allocates S3 (detection) = 90 ms and S4 (depth) = 150 ms on
> the *phone* reference devices (iPhone 13 / Pixel 6a). This Mac (Apple M5 Pro) is
> desktop-class silicon with no Neural Engine involved either way, so both inference
> numbers are predicted lower than the phone budget — detection because `yolo11n` int8 at
> 640×640 is small, depth because a ViT-S/14 forward pass is still real compute even on a
> fast CPU. Cold load — detection predicted from the 2.90 MiB artifact size; cold load —
> depth predicted much higher because ExecuTorch's `loadModel` compiles the whole graph
> synchronously and the artifact is 24.35 MiB, 8.4× the detector's. Render is predicted
> above the architecture budget's 30 ms because this is a **debug** build with Metro
> attached, not the release build §6.1 budgets for. Detection inference and its cold load
> are predicted anyway, as the exercise asks, even though the known `Invoke()` blocker
> (see `app/src/inference/decode.ts`'s header) means the actual value for **both** will be
> reported as N/A, not measured.

## Measurement conditions

**Every row below is a SIMULATOR measurement. None of it is a device number, and none of
it may be read as one.** The iOS Simulator runs the app as a native macOS process on this
Mac's own CPU — there is no Neural Engine, no mobile thermal envelope, no mobile memory
ceiling, and (for XNNPACK/CPU-backed ops) no phone-class core. It is faster than a phone
at some phases (desktop-class single-thread performance) and would be irrelevant-to-absent
at others (any CoreML/ANE path, which this configuration does not use). Run B is left
blank rather than filled with a guess — no device run exists yet; see "What was not
measured."

| | Run A | Run B |
|---|---|---|
| Device or simulator, exactly | iOS Simulator (iPhone 17, iOS 27.0) on this Mac (Apple M5 Pro, MacBook Pro, model `Mac17,8`, 18 cores / 24 GB) | *(no device run this round)* |
| OS version | iOS 27.0 (simulator runtime); host macOS 26.6.2, build 25G83 | |
| Build type | **debug** — Expo dev client (`npx expo run:ios`'s Debug configuration), not release | |
| JS dev server attached | **yes** — Metro (`npx expo start --dev-client`), on `localhost:8081` | |
| Detection backend | **N/A — never reached.** `analyzeImageUri` does not call `loadDetectorModel()` or any `model.run()`/`runSync()` on this path at all (see the "Cold model load" and "Per-image latency" sections below for the full explanation); even if it were wired up, `Invoke()` is structurally broken on this toolchain/runtime combination per `app/src/inference/decode.ts`'s header | |
| Depth backend | **XNNPACK**, confirmed from the export pipeline, not a live backend-selection log line: `docs/model-card-v1.md` records `depth.pte` as lowered via `to_edge_transform_and_lower(..., XnnpackPartitioner())` — the backend is baked into the artifact at export time, not chosen at load time, so there is no separate runtime log to point at | |
| Image used | `round4-landscape-4x3.jpg`, 707×530 source JPEG (`app/assets/test-images/`) — resized by `decodeImage` to SOURCE long-side 640 before any inference sees it. Fixed for all 25 warm runs (see judgment call below) | |
| Warm runs, n | 25 (+ 1 discarded warm-up pass, to let the depth model's cache populate before any sample counts) | |
| Date | 2026-10-22 | |

> **Judgment call — which bundled image.** `BundledPhotosCaptureSource.captureStill()`
> round-robins through 4 bundled photos on every call. The benchmark harness calls it
> **once**, before the loop, and reuses that single URI for all 25 iterations — so "same
> bundled image" held, but which of the 4 images that is depends on how many times the
> source was called earlier in the session (e.g. once more if "Use bundled test photo" was
> pressed first). Confirmed from the Metro log which image this run actually used
> (`round4-landscape-4x3.jpg`); not assumed from the array's first entry.

## Cold model load — reported separately

Happens once per launch. The user watches it. It is not part of the per-image distribution
and averaging it in corrupts both numbers.

| Artifact | Load time | n |
|---|---|---|
| Detection `.tflite` | **N/A — not measured.** `loadDetectorModel()` is defined in `app/src/inference/detector.ts` but is never called anywhere in the current app (`grep` confirms zero call sites outside its own module) — the pipeline bypasses it entirely in favor of the bundled raw-tensor fixture. There is no cold-load event to time on this build, separate from (and in addition to) the known `Invoke()` blocker | 0 |
| Depth `.pte` | 45 ms, 51 ms, 58 ms across 3 fresh process launches — **median 51 ms** | 3 |
| **Total, first usable frame** | **51 ms** (depth only — detection load does not happen on this path, so this total understates what "first usable frame" will cost once detection is wired up for real) | n/a |

## Per-image latency, warm

> **Preprocess is reported as two sub-phases, not folded into one number.** The first
> instrumented pass combined decode and the detector's letterbox under one timer and
> produced a nonsensical-looking p50/p95 spread (p50 62 ms, p95 399 ms from 50 pooled
> samples) — splitting them showed why: two very differently-costed steps were sharing one
> array, and the percentile of the *pool* isn't the percentile of either step. Split, both
> are internally consistent (tight p50/p95 gaps). The template's single "Preprocess" row
> below is the sum of the two, with the split shown so the dominant sub-step is visible
> rather than hidden — exactly the failure mode this report's own instructions warn against
> for phases in general.

| Phase | p50 | p95 | Notes |
|---|---|---|---|
| Preprocess | **452 ms** | **460 ms** | Decode = 392/399 ms (native resize + **pure-JS** `jpeg-js` decode, on the JS thread) + detector letterbox/normalize = 60/61 ms (JS, pure arithmetic). Decode is ~87% of this row |
| Detection inference | **N/A — not measured** | **N/A — not measured** | No `model.run()`/`runSync()` call exists on this path (bundled fixture tensor substitutes for it — see `analyzeImage.ts`'s header); `Invoke()` is independently known-broken. Reporting 0 here would read as "measured, and fast," which is false |
| Decode + NMS | 2 ms | 2 ms | In TypeScript, over the fixture tensor's 8400 anchors. Confirmed NOT significant, as the template's own analysis row predicts it might not be |
| Depth inference | **344 ms** | **360 ms** | Real, on-device `execute('forward')` only — depth's own letterbox (39/41 ms) is excluded from this row and counted in Preprocess above instead, so this is purely the ViT-S/14 forward pass |
| Fusion | **56 ms** | **61 ms** | Per-kept-detection: un-letterbox to SOURCE, re-letterbox into DEPTH_INPUT, `medianInRegion` (full sort of every pixel in the box's depth-space rect). Not the "almost free" arithmetic it reads as in `analyzeImage.ts` — the sort is the cost |
| Render | 28 ms | 33 ms | Proxy: time from `setResult(...)` commit to the next **painted** frame, via double `requestAnimationFrame`. Measures only the JS-triggered re-render + draw, not image-selection-to-first-paint |
| **Total** | **~921 ms** | **~957 ms** | Sum of the measurable rows above (excludes Detection inference, which has no value to sum). **Not an N1-comparable number** — debug build, simulator, and missing a real phase |

## Memory

**Method:** `ps -o rss= -p <pid>` against the Simulator app's own macOS process (the iOS
Simulator runs the app as a native Mac process, visible to `ps` like any other), polled at
roughly 1.5 s intervals through a fresh app launch → one analysis (loads both models) →
the 25-run benchmark. This is a coarse, polling-based **proxy**, not Xcode's own memory
gauge (not scriptable here) and not an Instruments trace (not run this round) — a real
spike between two 1.5 s samples would be missed, and RSS on macOS counts some pages (e.g.
shared libraries) that a phone's accounting would charge differently. Stated honestly as
the limitation it is, not smoothed over.

| Measurement | Value | Method |
|---|---|---|
| Baseline, no models loaded | 556.7 MB | `ps -o rss=`, fresh launch, before any button press |
| Both models loaded, idle | 876.8 MB | `ps -o rss=`, ~6 s after one "Use bundled test photo" run, RSS stable across 4 more samples |
| **Peak during inference** | **911.1 MB** | `ps -o rss=`, highest of ~15 samples at 1.5 s intervals through the 25-run benchmark |

> This is the figure that decides which phones are excluded, and it is the one nobody
> measures until a crash report arrives. Two runtimes hold their own weights and their own
> intermediate tensors, and they do not share an allocator.

## Comparison with Lesson 04

**Not available — this section cannot be filled in honestly.** `docs/execution-target-comparison.md`
(Lesson 04, step 7's own deliverable for exactly this comparison) is, as of this report,
still the **unfilled prompt text itself** — byte-for-byte the same content as
`lessons/05-mobile-client-and-delivery/resources/prompts/08-latency-measurement.md`'s
sibling prompt file, confirmed by reading both. No Round 2 measurement, no Round 3 table,
and no Python reference number were ever recorded into it. `docs/evaluation-v1.md` and
`docs/roadmap.md` were also checked — neither records a workstation/fp32 latency figure;
`docs/roadmap.md` line 146 lists "Workstation baseline timings" as a **planned** W4 input,
not a result. This is a real gap in Lesson 04's own deliverables, not a gap this round
could close: the brief for this step asks to *read* Lesson 04's recorded number, not
re-measure Python, and there is no recorded number to read.

| Measurement | Value | What it tells you |
|---|---|---|
| Python reference, this machine | **not recorded anywhere in this repo** | — |
| On-device p95 | 957 ms (simulator, debug — see caveats above; excludes the unmeasured Detection inference phase) | How fast the product is, on this one incomplete measurement |

The gap this round can state is prediction-vs-measurement (below), not Python-vs-device.

## Analysis

**Dominant phase:** `Preprocess`, **52%** of the measurable p95 (460 ms of 957 ms). Within
that, the **decode** sub-step alone (JPEG decode + native resize, on the JS thread) is
**42%** of total p95 by itself — larger than Depth inference (38%), the phase every prior
document in this project (the architecture budget, Round 1's own prediction) expected to
dominate.

**What code can actually change:**

| Phase | Changeable? | Lever |
|---|---|---|
| Cold load | Partly | Depth is already fast (51 ms median) — no action needed there. Detection's cold load is currently moot: nothing calls it |
| Preprocess | **Yes — the clearest lever in this report** | The 392-399 ms decode cost is `jpeg-js`, a pure-JS JPEG decoder, running on the JS thread (`app/src/inference/image.ts`). A native decode (off-thread, hardware-accelerated) is very likely the single highest-leverage change available to this app right now |
| Detection | Unknown | Cannot be assessed — no real inference exists to measure or improve yet. Fixing `Invoke()` is the prerequisite to even asking this question |
| Depth | Partly | 344-360 ms execute() is real ViT compute. `docs/architecture.md` §6.2's own escape lever (`DEPTH_SIZE` 518→256) is the main option; a smaller checkpoint is the other, more disruptive one |
| Decode + NMS | Checked, not significant | 2 ms. Confirms the template's own suspicion — this was never the cost anyone should have worried about |
| Fusion | Yes, and worth doing | 56-61 ms, far above what the "mostly arithmetic" framing in `analyzeImage.ts`'s own code implies. `medianInRegion`'s full `Array.sort()` per detection is the cost; a partial-selection (quickselect) or subsampling the region would cut it |
| Render | Marginal | 28-33 ms, roughly in line with the architecture budget's 30 ms despite being a debug build. Not where the problem is |

**Prediction versus measurement:** They largely did **not** agree, and the gap is mostly
in one direction — real compute costs were underestimated, model-loading costs were
overestimated:

- **Depth cold load:** predicted 1500 ms, measured 51 ms — **29× over**-predicted. The
  24.35 MiB artifact's size does not predict ExecuTorch `loadModel` time the way it was
  assumed to; the quantized, pre-lowered graph loads fast.
- **Preprocess:** predicted 60 ms, measured 452-460 ms — **~8× under**-predicted, entirely
  because the prediction didn't anticipate that JPEG decoding would be done in pure JS
  rather than natively. This is the single biggest miss in the report and the most useful
  one — it is a real, fixable architectural choice, not an inherent cost.
- **Depth inference:** predicted 80 ms, measured 344-360 ms — **~4.3× under**-predicted.
  The architecture budget's own S4=150 ms phone allocation was a better anchor than the
  "desktop CPU should be faster" reasoning used in Round 1; a ViT-S/14 forward pass is
  real compute regardless of core speed.
- **Fusion:** predicted 3 ms, measured 56-61 ms — **~19× under**-predicted. The prediction
  treated `medianInRegion` as free arithmetic; it is a full sort over every pixel in a
  region, which is not free.
- **Render:** predicted 50 ms, measured 28-33 ms — over-predicted by roughly 1.6×. The
  "debug build is slow" intuition was directionally right elsewhere but not here.
- **Peak memory:** predicted 350 MB, measured up to 911 MB — **~2.6× under**-predicted.

## Recommendation for N1

**No credible N1 recommendation can be made from this round.** Two of N1's own stated
conditions (`release` build, a named physical device) are both absent here, and the
phase most likely to be the second-largest real cost once unblocked — Detection
inference — has no measured value at all. What this round *does* support: the
preprocess-decode cost (392-399 ms, simulator, debug) **on its own already exceeds N1's
entire 400 ms budget**, before either model has run. Moving JPEG decode off the JS thread
is very likely a prerequisite for N1 to be reachable at all, independent of anything
found about the two models.

**Cold start:** Excluded from N1 per `docs/requirements.md` (N6 covers it separately, ≤3.0 s
target) — consistent with this report. Depth's measured 51 ms median is comfortably inside
N6; detection's contribution to N6 is unknown, since it is not exercised in this build.

## What was not measured

- **Any physical device.** N1's own named reference devices (iPhone 13 / A15, Pixel 6a /
  Tensor G1) — not available this round per this task's own scope (simulator only); see
  the mobile-mcp physical-device automation blocker noted for this session.
- **A release build.** Everything above is debug, with Metro attached. Release is very
  likely faster on every JS-thread-bound phase (preprocess decode, fusion) and possibly
  on native-call overhead too — untested.
- **Detection inference**, at all — the `Invoke()` blocker means no real detection latency
  number exists anywhere, on any device, in this project, today.
- **Other images.** All 25 warm runs used one fixed image (707×530, a mid-complexity
  furniture scene) — a busier or emptier scene, or the other 3 bundled aspect ratios, were
  not exercised. `medianInRegion`'s cost in particular likely scales with detection count
  and box size, so Fusion's 56-61 ms is this-image-specific, not a project-wide constant.
- **Sustained use / thermal throttling.** 25 back-to-back runs in ~15-20 s is not N5's
  "sustained throughput" bar, and the Simulator runs on this Mac's own cooling, which says
  nothing about a phone's thermal envelope.
- **Low-battery states**, Android, or anything about NNAPI/CoreML/GPU delegates — none of
  those paths are reachable from a debug iOS Simulator build.

**A reader should not conclude** that this app meets, or is close to meeting, N1. The one
number this report licenses with confidence is the **shape** of the bottleneck — JS-thread
image decode is a larger, more fixable cost than either model — not a pass/fail verdict on
the 400 ms target, which this round was never positioned to render.
