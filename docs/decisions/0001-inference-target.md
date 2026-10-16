# ADR 0001 — Inference target: on-device

- **Status:** accepted
- **Date:** 2026-09-01
- **Deciders:** Project team (Lesson 01, step 8)

## Context

The Smart Scene Analyzer runs two models per image — YOLO11 for detection and Depth
Anything V2 for monocular depth — and returns boxes with a per-object relative depth score
to someone holding a phone. Where those two models execute is the first architectural
decision, and every subsequent one inherits it: the latency budget's largest term, whether
image bytes leave the handset, what a model update costs to ship, and whether there is a
bill that scales with usage.

Three properties forced the decision rather than left it to taste:

1. **The client is a real application.** Lesson 05 builds an Expo app for iOS and Android.
   The consumer is a camera-shaped UI, not a `curl` loop, and a camera-shaped UI makes far
   more requests than a benchmark does.
2. **The p95 latency target is 400 ms** (`docs/requirements.md`, N1). On a served
   architecture, a 250 KB upload over a 5 Mbit/s link spends roughly 400 ms in transit
   before the server has seen a byte — the entire budget, on physics we do not control,
   and worse on a slower link.
3. **The project has a hard 20-credit Roboflow budget for its whole lifetime**
   (`docs/credit-budget.md`), and no top-up. A per-request billing model is not something
   this project can absorb being wrong about.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| **A — Cloud, self-hosted.** Our container serves both models | One codebase and one architecture; no export toolchain; no per-platform numerical validation; models update by deploying; no device floor | A network round-trip inside a 400 ms budget, which transit alone can exhaust; a GPU or a slow CPU container to size and pay for; image bytes leave the device, making retention a policy rather than a fact; useless offline |
| **B — Cloud, hosted API.** Detection served by Roboflow | No weights in our image; no serving infrastructure at all; fastest to stand up | **Bills per request**, and a camera client generates requests at a rate no `curl` loop predicts — the one option whose cost scales with success. Against a 20-credit lifetime budget this is an uncapped liability. Still pays the transit cost of A, and still cannot work offline |
| **C — On-device.** TFLite for detection, ExecuTorch for depth | No round-trip; the largest latency term is deleted rather than optimized; works in airplane mode; image bytes cannot leave, so privacy is structural rather than promised; zero marginal cost per inference; zero Roboflow credits at inference time, by construction | Forces per-platform export, int8 quantization, and numerical parity validation of two models against the reference; ships two native runtimes and tens of MB of artifacts; sets a device floor that excludes some phones; a model fix becomes an app-store release, not a deploy |

## Decision

**Both models run on the handset.** Detection is YOLO11 exported to int8 TFLite via
`react-native-fast-tflite`; depth is Depth Anything V2 exported to ExecuTorch (`.pte`) via
`react-native-executorch`. No network call exists anywhere on the inference path.

Option B was rejected first and hardest: it is the option that looks cheapest while being
the only one whose cost grows with how much anyone uses the app. Option A remained
plausible until the latency arithmetic — transit alone can consume the whole of N1 — made
its budget unpassable on a link we do not control.

The honest summary is that C is not the cheap option. It trades a hosting bill for an
export toolchain, two native runtimes, and a release cycle per model update. It is
expensive in a different currency, and naming the currency is most of what this record is
for.

## Consequences

**Easier:** N1 has no transit term, so the budget is entirely compute we control. N1a
(bytes leaving the device) is 0 by construction. N14 (image retention) is answered by
there being nowhere to send an image. Roboflow credits at inference time are structurally
zero — the app ships with no credentials, so the call cannot be made even by mistake.

**Harder:** every model change now crosses an export boundary and must be proved
numerically equivalent to the reference. The fusion algorithm exists twice, in Python and
TypeScript, and only a shared fixture set keeps the two honest. Shipping a fix means an
app-store review.

**Confirmed harder, Lesson 04 (2026-10-16) — not predicted anymore, measured:**

- **The export toolchain churned out from under the plan.** `ultralytics` deprecated
  `format='tflite'`/`int8=True` mid-release (now `format='litert'`/`quantize=8`), and the
  dependency list researched from documentation (`tensorflow`, `onnx2tf`, ...) was wrong
  for the version actually installed — the real requirement (`litert-torch`,
  `ai-edge-litert`) was only found by running the export and reading the failure. This is
  exactly the "an export toolchain" cost named above, now with a specific shape: it is not
  a one-time cost, it is a target that moves under a pinned dependency.
- **One of the two models could not be exported at all.** `optimum-executorch` 1.1.0 has
  no `depth-estimation` task registered — not a quantization or accuracy failure, a
  missing capability in the library this plan depends on. The fusion-layer duplication
  cost above assumed both sides would exist to duplicate; right now only detection does.
- **Quantization is not free on the classes that were already weak.** `storage`'s mean
  confidence dropped 0.133 under int8 — small in isolation, but stacked on a class that
  was already this model's second-weakest (`docs/evaluation-v1.md`). The export cost is
  not evenly distributed across classes, and nothing in the original Context anticipated
  that a fixed per-model cost would land harder on specific, already-fragile classes.

**What now depends on this decision:**

- `src/smart_scene_analyzer/` — the export pipeline and the reference implementation the
  device is checked against
- `app/` in its entirety, including `app/src/fusion/` as a port of the Python reference
- `app/assets/models/` — the artifact contract: tensor shapes, output order, label order,
  normalization constants, all fixed by the export and assumed by the app
- `docs/artifact-budget.md` — a budget that exists only because we ship artifacts; it does
  not bill, it fails at install time
- `.claude/hooks/artifact_drift.py` — guards the seam this decision created
- `.claude/hooks/units_guard.py` extending to `app/` — depth is now *computed* on the
  device, not merely displayed there
- Lessons 04 and 05, which are written for this path

**What this forecloses:** server-side batching, model updates without a release, and any
architecture that puts a request on the inference path. Adding one back is not a change of
implementation; it is a reversal of this ADR.

## Amendment — Lesson 04 measurements (2026-10-16)

The options above were argued from latency arithmetic and credit economics (ADR 0001's
original context). This project has since actually exported, quantized, and measured one
side of that argument — the numbers below replace assumption with evidence, on the
detection model; the depth model's export failed outright, which is itself a finding.

**Artifact sizes** (`docs/artifact-budget.md`, 2026-10-15):

| Artifact | Size | Status |
|---|---|---|
| Detection (`detector.tflite`, int8) | **2.90 MiB** | Exported successfully |
| Depth (`depth.pte`, ExecuTorch) | — | **FAILED.** `optimum-executorch` 1.1.0 has no `depth-estimation` export task registered — confirmed from its source, not inferred from the error alone. Not a quantization or accuracy problem; a toolchain gap. See `docs/model-card-v1.md`'s Depth "Exported artifact" section |

Only one of the two planned on-device artifacts exists. N9a's 40 MB combined budget is
therefore unverified on its real constraint — comparing two artifacts — because there is
only one to weigh.

**Export parity** (`tests/test_export_parity.py`, 8 held-out test-split images, 2026-10-15):

| Field | Value |
|---|---|
| Match criterion (tolerance) | Same class, IoU ≥ 0.5 — accepted *before* the run, not fit to the result |
| Match rate | 34/37 = 91.9% |
| Worst per-class confidence delta | `storage`, **−0.133** (quantization shifted this already-weak class's confidence down further; `door` moved the other way, +0.123 — evidence the shift is quantization noise, not one-directional) |
| Mean IoU, every class | ≥ 0.866 — box geometry well preserved |
| Depth parity | **Not measurable** — no `.pte` artifact exists to compare |

**Default execution backend per platform** (`docs/execution-target-comparison.md`,
2026-10-15, every availability claim cited with a URL and check date):

| Platform | Default | Why not the alternatives |
|---|---|---|
| Android | **XNNPACK/CPU** | NNAPI is deprecated as of Android 15 (Google's own guidance now points to GPU/CPU); GPU delegate is plausible on paper but unmeasured on any real device from this project |
| iOS | **XNNPACK/CPU** | CoreML/ANE **cannot run this artifact at all** — the delegate only supports float32/FP16, and this artifact has int8-quantized internals; GPU/Metal's behavior (including Simulator support) is undocumented |

Measured latency: **11.47 ms p50 / 11.69 ms p95** for detection, XNNPACK/CPU, n=30 — on
this development machine (Apple M5 Pro), explicitly **not** a phone measurement. Every
phone-specific row (real device CPU, NNAPI, CoreML, GPU delegate) is marked UNMEASURED in
`docs/execution-target-comparison.md`, not estimated.

**What this amendment does not change.** The original Decision stands: both models still
target on-device execution, for the same latency-budget and credit-liability reasons in
the original Context. Nothing measured here argues for moving inference back to the cloud
— see that document's own Round 4, which works through exactly this question and finds no
measurement in the table with a path back to Option A or B. What changed is confidence:
the detection half of the decision is now backed by a real artifact and a real parity
number instead of an estimate; the depth half is not yet backed by anything, because it
does not yet exist.

## Revisit when

Any of the following becomes true:

- The combined size of two native runtimes plus two artifacts proves unaffordable against
  `docs/artifact-budget.md`, **and** consolidating onto a single runtime does not recover
  it. (Consolidation is the first fallback, not a return to the cloud.) **Partially
  unmeasurable right now**: only the detection artifact's size (2.90 MiB) is known; the
  depth side of this comparison cannot be evaluated until an export exists.
- Peak RSS with both models loaded excludes a device class the product must support.
  **Still open** — requires Lesson 05's on-device measurement; nothing in Lesson 04 could
  measure this on a development machine.
- A required capability genuinely cannot run on-device — a model too large to quantize
  without unacceptable accuracy loss, for instance. **Not yet triggered, but the closest
  live example of what this looks like**: the depth export's current failure is a
  toolchain-support gap (`optimum-executorch` has no depth-estimation task), not a proof
  that Depth Anything V2 cannot run on-device at all — a lower-level export path direct on
  ExecuTorch's core APIs remains untried. If that path is attempted and *also* fails for a
  capability reason (not a library-support reason), **that** would trigger this clause.
  Confusing a toolchain gap for a capability gap here would be the wrong kind of mistake —
  it would abandon on-device depth over a library version, not a real constraint.

Cost or convenience alone is not a trigger. The offline and privacy properties above are
load-bearing product behaviour, not implementation detail.

Lesson 05 measures on-device latency and peak RSS on real hardware — the two rows above
still marked open. That measurement, not this amendment, is what actually closes or
triggers the first two bullets.
