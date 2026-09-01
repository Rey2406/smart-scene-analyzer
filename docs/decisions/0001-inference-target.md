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

## Revisit when

Any of the following becomes true:

- The combined size of two native runtimes plus two artifacts proves unaffordable against
  `docs/artifact-budget.md`, **and** consolidating onto a single runtime does not recover
  it. (Consolidation is the first fallback, not a return to the cloud.)
- Peak RSS with both models loaded excludes a device class the product must support.
- A required capability genuinely cannot run on-device — a model too large to quantize
  without unacceptable accuracy loss, for instance.

Cost or convenience alone is not a trigger. The offline and privacy properties above are
load-bearing product behaviour, not implementation detail.
