# Artifact budget

**Target: the app installs and runs on a mid-range phone from three years ago.**

This is the project's second budget, and it is the third kind of constraint the course
teaches. It is worth being precise about how it differs, because the difference is the
reason it exists.

| | Roboflow credits | The artifact budget |
|---|---|---|
| Currency | Prepaid credits | None. Bytes and milliseconds |
| How you exceed it | Spend more than you have | Ship a file that is too big |
| What happens | The operation **fails**. `credit_gate.py` refuses the call | The build fails, or the app installs and crashes **on somebody else's device** |
| Who tells you | A hook, before the money moves | Nobody. You find out from a bug report |
| Can a hook enforce it? | Yes, and one does | **No.** There is no tool call to intercept |

A hard cap enforces itself. A meter at least invoices you. This budget does neither: it is
enforced by a store's download threshold, a device's memory, and a user's patience — none
of which are in this repository. The only defence is to measure early and write it down,
which is what this file is for.

## What this project ships

Two model artifacts **and two native ML runtimes**. That combination is a deliberate
consequence of running detection on TFLite and depth on ExecuTorch, and it is the single
largest line in this budget. Record it honestly rather than discovering it at submission.

| Component | What it is | Measure with |
|---|---|---|
| Detection model | YOLO11 int8 `.tflite` | `ls -l app/assets/models/*.tflite` |
| Depth model | Depth Anything V2 `.pte` | `ls -l app/assets/models/*.pte` |
| TFLite runtime | `react-native-fast-tflite` native libs | Per-ABI, from the build output |
| ExecuTorch runtime | `react-native-executorch` native libs | Per-ABI, from the build output |
| JS bundle + assets | Everything else | Metro bundle output |

## Upstream checkpoints — measured 2026-10-01

These are the **pre-export** checkpoints pulled in Lesson 03, not the artifacts the app
ships. They are recorded because they are the first real numbers this budget has had, and
because the ratio between them is the finding.

| Model | Params | On disk | Stored as | int8 projection |
|---|---|---|---|---|
| YOLO11n | 2,624,080 | 5.35 MiB (`yolo11n.pt`) | fp16 | **~2.50 MiB** |
| Depth Anything V2 Small | 24,785,089 | 94.59 MiB (HF cache) | fp32 | **~23.64 MiB** |
| **Combined weights** | 27,409,169 | ~100 MiB | | **~26.1 MiB** |

**The depth model is 9.4× the detection model.** Nothing in this project's framing
anticipated that — detection is the headline feature and depth is the second signal, but
depth is where essentially all of the bytes are. Both `⚠️ OPEN` items below should be read
with that in mind: if the artifact budget fails, it fails on the depth model, and
consolidating onto one runtime is a question about *which* runtime can carry a 24.8M-param
ViT, not about the detector.

**The int8 column is a projection, not a measurement.** It assumes one byte per parameter
and ignores quantization metadata, per-channel scales, and the graph itself. Treat it as a
floor. The real number comes from `ls -l app/assets/models/` after the export, and belongs
in the Budget table below — not here.

Note the YOLO11n checkpoint is **fp16 on disk**, which is why 5.35 MiB is smaller than the
10.01 MiB its parameter count implies at fp32. Comparing checkpoint file sizes across
models is comparing storage formats, not model sizes; the parameter count is the
comparable figure.

Not counted here, and larger than either model on some platforms: the two runtimes'
**native libraries**, per ABI. Those arrive at the first `expo run:*` and are the line most
likely to decide the device floor.

## Budget

Fill the targets in before the first build, not after. A target chosen after seeing the
number is not a budget, it is a description.

| Line | Target | Measured | Date | Notes |
|---|---|---|---|---|
| Detection model | *(no target was pre-stated — not filled in before this measurement, so none is claimed now)* | **2.90 MiB** (int8) | 2026-10-15 | `v1-baseline` → `app/assets/models/detector.tflite`. **fp32 size not yet produced** — this run only exported int8, so the quantization delta this row asks for is still missing; needs a follow-up fp32 export to compare against |
| Depth model | | **FAILED** | 2026-10-15 | Export attempted, not produced — `optimum-executorch` 1.1.0 has no `depth-estimation` export task registered. See `docs/model-card-v1.md`'s Depth "Exported artifact" section for the confirmed root cause |
| **Bundled models, total** | | | | |
| Android install size | | | | From the Play Console or `bundletool` |
| iOS install size | | | | From App Store Connect or a local `.ipa` |
| Peak RSS, both models loaded | | | | The number that decides which phones are excluded |
| Cold model load, both | | | | Seconds. The user is staring at a spinner for this |

### How to measure install size honestly

An APK on disk is not what a user downloads, and neither is an `.ipa`. Android splits by
ABI and density; iOS thins per device and re-signs. Measure the **delivered** size, from
the store's own reporting or from `bundletool build-apks --connected-device`, and say
which method produced the number. A figure whose method is not stated is not comparable to
the next one.

### The measurement that decides the device floor

Peak resident memory with both models loaded, on the smallest device you intend to
support. Two runtimes each hold their own weights and their own intermediate tensors, and
they do not share an allocator. This is what actually excludes a phone — not the download
size, which merely annoys people.

## What is not in this budget

| Item | Why not |
|---|---|
| Server cost | There is no server |
| Per-inference cost | Inference runs on hardware the user already paid for |
| Network egress | Nothing is transferred on the inference path |
| Roboflow credits at inference time | Zero, structurally — the app has no credentials |

That last row is worth pausing on. Under the previous cloud architecture, "zero Roboflow
credits on the request path" was a design commitment somebody could break with one import.
It is now a property of the artifact: the app ships with no API key and no network code on
that path, so the call cannot be made. Constraints enforced by structure outlive
constraints enforced by discipline.

## ⚠️ OPEN — resolve before running a cohort

- **The device floor has not been chosen.** `react-native-executorch` requires iOS 17+ and
  Android 13+; that is the runtime's floor, not necessarily this app's. The memory ceiling
  above will move it upward, and nobody has measured it yet.
- **Whether two runtimes are affordable at all.** If the combined native library size
  proves unacceptable, the fix is to consolidate onto one runtime — either YOLO11 to
  `.pte` behind a wrapper module, or Depth Anything V2 to TFLite. Both are unverified, and
  both are a real workstream. Measure before assuming this is fine.

  **Updated 2026-10-01 with the first measurement.** The checkpoints above change which
  direction to consider first. Depth Anything V2 Small is 24.8M parameters against YOLO11n's
  2.6M, so a consolidation that moves the *detector* onto ExecuTorch relocates 9% of the
  weights, while one that moves the *depth model* onto TFLite relocates 91%. If
  consolidation happens, moving YOLO11 to `.pte` is the smaller, lower-risk change — and it
  leaves the 24.8M-param ViT on the runtime that was chosen for it. This is a hypothesis
  from parameter counts, not a measurement of either runtime's native library size, which
  remains the actual unknown.

## Reconciliation

At the end of every lesson that changes a model or adds a dependency with native code:

1. Re-measure every line above that could have moved.
2. Record the new figure with its date — keep the old row rather than overwriting it. The
   trend is the finding; a single number is not.
3. If a line moved and you cannot say which change moved it, that is the thing to
   investigate before adding anything else.
