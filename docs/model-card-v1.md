# Model card — `v1-baseline`

One card per trained model. A model referenced anywhere in this project without a card is
a model nobody can reason about in three months.

## Identity

| Field | Value |
|---|---|
| Model name | `v1-baseline` |
| Architecture | `yolo11n` |
| Task | Object detection |
| Base checkpoint | `yolo11n.pt` (COCO) |
| Trained on | Dataset **v1** — ⚠️ a *local* build, not a Roboflow version. See the note below |
| MLflow run | `v1-baseline`, experiment `smart-scene-analyzer-detection` |
| Weights | `runs/detect/runs/v1-baseline/weights/best.pt` — not committed. The doubled path is ultralytics prepending its own `runs/detect` settings prefix to `--project runs/` |
| Git commit | `d115a49` — **working tree dirty**; the run is tagged `git_dirty=true` |
| Trained | 2026-10-07, local (Apple M5 Pro, `mps`) |
| Credits consumed | **0** |

> ⚠️ **The dataset reference is weaker than this row usually implies.** `data/v1` is not an
> immutable Roboflow snapshot — the workspace reached zero credits before a version could
> be generated, so no `versions_generate` ever succeeded. Provenance is instead:
> `scripts/make_local_version.py` at seed 1, over `build/yolo-sunrgbd/` from
> `scripts/convert_sunrgbd.py`, against `docs/taxonomy.md` (K=10, alphabetical, ADR 0003).
> Reproducible, but by convention and git history rather than by platform guarantee.
> The MLflow run records this in `dataset_provenance` rather than logging a bare "v1".

> ⚠️ **The git tree was dirty at training time**, so the commit does not fully describe the
> code that produced these weights. Recorded rather than smoothed over.

## Exported artifact

> **Lesson 04, step 2.** Every value below is **INTENDED** — what this export is designed
> to produce, written before the export runs. Step 4 replaces each with what the export
> **actually** produced; where they differ, the artifact is right and this table was wrong,
> and the INTENDED marking comes off only after that reconciliation. A row left blank here
> is a to-do; a row guessed after seeing real export output is a lie the app will act on.

### Detection — `v1-baseline` → `detector.tflite`

> **Reconciled against the INTENDED table below it (Lesson 04, step 4, 2026-10-15).**
> Per every row: **matches** / **differs** / **was not knowable in advance**. Where the
> artifact and the INTENDED guess differ, the artifact is right and the guess was wrong —
> this project's own rule (`docs/architecture.md` §4) is that nothing checks the contract
> at runtime, so a wrong assumption here is exactly the kind of bug that ships silently.

| Field | **Actual** (read from `app/assets/models/model_contract.json`) | vs. INTENDED |
|---|---|---|
| Format | LiteRT int8 (`litert_torch` path, ultralytics ≥8.4.118's current export — `format='tflite'`/`int8=True` are deprecated aliases for this) | **differs** — INTENDED said "int8 TFLite" naming only; the underlying toolchain (onnx2tf vs. `litert_torch`) wasn't knowable until run |
| Input tensor | `[1, 3, 640, 640]`, **float32**, **NCHW** | **DIFFERS, materially.** INTENDED assumed `[1, 640, 640, 3]` uint8 NHWC, following `docs/architecture.md`'s C3. The actual export preserves PyTorch's native NCHW layout and a float32 I/O boundary — three wrong assumptions at once (layout, dtype, and therefore normalization scheme) |
| Normalization | `[0, 1]` float (divide by 255, no further scaling) | **differs** — INTENDED assumed raw uint8 pixels quantized directly at the boundary; the actual input is a plain `[0,1]`-scaled float, with int8 quantization applied *internally* to weights/activations rather than at the input tensor |
| Quantization (input tensor) | **None** — the input tensor itself carries no scale/zero_point; `quantize=8` quantizes the graph's internal weights and activations, not its I/O boundary | **differs** — INTENDED expected scale/zero_point on the input itself |
| Letterbox pad value | `(114, 114, 114)` | matches (A4) — unaffected by the layout/dtype difference above, since padding is a pixel-space concept applied before the tensor is built |
| Output tensors | **One** tensor, `[1, 14, 8400]` float32: 4 box channels (`cxcywh`, confirmed **normalized** to `[0,1]`) + 10 class-confidence channels, 8400 anchor positions | **was not knowable in advance**, now resolved. Determined empirically on a real calibration image: `cxcywh` interpretation gave a 100% valid-box rate vs. 0% for `xyxy` (unambiguous); coordinate 95th-percentile of 0.89 (≤2.0 threshold) confirmed normalized rather than 640-px pixel space |
| Label order | `bed, bin, chair, door, lamp, pillow, screen, sofa, storage, table` | **matches exactly.** Confirmed independently via `YOLO(best.pt).names`, index-for-index against `docs/taxonomy.md` |
| Default thresholds | confidence 0.25, IoU 0.45 | matches — these are this project's own chosen defaults (A8), written into the manifest by the exporter, not read back from the file |
| Max detections | 100 per image | matches — same caveat: an exporter-assigned default (A7), not an empirically-discovered artifact property |
| Calibration set | `data/v1/data.yaml`'s **val** split, 1,545 images | matches the INTENDED decision. Confirmed optimistic-parity caveat still applies to step 6 |
| sha256 | `3f5bf682...3ad83d` | n/a — INTENDED had no prediction here; now the artifact's fixed identity for `artifact_drift.py` |
| Size | 2.90 MiB | n/a — not predicted in INTENDED; comfortably inside N9a's 40 MB combined budget on its own |

**What this means for the app.** `docs/architecture.md`'s C3 contract — `[1, DET_SIZE,
DET_SIZE, 3]` NHWC with export-owned quantization — describes the *wrong* tensor for this
specific artifact. Any TypeScript code written against C3 as currently stated would build
a `[1,640,640,3]` uint8 buffer and feed it to a model expecting `[1,3,640,640]` float32 —
Round 1 of step 2's own exercise names the exact symptom: output runs, produces plausible
numbers, and is silently wrong. **`docs/architecture.md` needs a follow-up correction**
(out of this step's scope, flagged here rather than silently fixed) before Lesson 05 writes
any app-side tensor-building code against it.

#### Export parity (Lesson 04, step 6, 2026-10-15)

`tests/test_export_parity.py`, 8 held-out **test**-split images (not the val images used
for int8 calibration — see `tests/fixtures/export_parity_images.json`). Compared against
the original PyTorch model directly (`ultralytics.YOLO(...).predict()`), never against the
export re-read in a second runtime — the exported side is decoded through this project's
own `inference.decode` + `inference.nms`, exercising the same path the app will eventually
run.

| Field | Value |
|---|---|
| Match criterion | Same class, IoU ≥ **0.5** (the standard COCO/Pascal VOC convention — not chosen to fit) |
| Overall match rate | **34/37 = 91.9%** of PyTorch detections found a same-class, IoU≥0.5 match in the export |
| Accepted tolerance | match rate ≥ 90%, worst per-class mean confidence delta < 0.5 — both set *before* this run, in the test file |

| Class | PyTorch | Exported | Matched | Mean IoU | Mean conf. delta |
|---|---:|---:|---:|---:|---:|
| bin | 1 | 1 | 1 | 0.954 | +0.016 |
| chair | 22 | 21 | 19 | 0.866 | −0.080 |
| door | 1 | 1 | 1 | 0.899 | +0.123 |
| sofa | 3 | 3 | 3 | 0.883 | +0.033 |
| **storage** | 2 | 2 | 2 | 0.967 | **−0.133** |
| table | 8 | 8 | 8 | 0.902 | −0.065 |

**Worst per-class confidence delta: `storage`, −0.133.** Box geometry is well preserved
everywhere (every class's mean IoU ≥ 0.866); the effect of quantization is almost entirely
on confidence, not position. `door`'s confidence *rose* under quantization (+0.123) —
evidence the shift is quantization noise in both directions, not a one-way degradation.
The 3 unmatched detections are concentrated in one image with a `chair`/`sofa` pair the
export reordered — the same confusion `docs/evaluation-v1.md`'s confusion matrix already
attributes to TAXONOMY (an ambiguous seating-capacity boundary), not something new this
export introduced.

**Depth parity: not possible to check.** The depth export failed (see below) — there is no
`.pte` to compare against, so `test_depth_parity_is_not_possible_yet` records that and
skips rather than fabricating a result.

**Ship verdict:** the **export is faithful enough to ship** — no class collapsed, IoU held
everywhere, and the confidence shift (worst −0.133, `storage`) is a real but bounded effect
of int8 quantization, not a correctness break. This verdict is scoped to the export step
only. It does **not** mean the underlying model is ready to ship: this is still the
interrupted 29-of-50-epoch checkpoint (`docs/evaluation-v1.md`), and `storage` was already
one of its weaker classes (P=0.606, R=0.510) before quantization shaved another 13 points
off its confidence — stacking a real effect on top of an existing weakness. The fix for
that is finishing training, not re-exporting; re-exporting a better checkpoint through this
same pipeline should be expected to reproduce a similarly small, acceptable parity gap.

### Depth — Depth Anything V2 Small → `depth.pte`

> **Export FAILED (Lesson 04, step 4, 2026-10-15). No `.pte` was produced — the INTENDED
> table below remains unreconciled and unverified.** This is the finding this step's own
> warning anticipated ("whether Depth Anything V2's ViT survives the ExecuTorch recipe...
> has not been confirmed"), and per that step's explicit instruction, no different
> checkpoint, recipe, or task was substituted to force a result.
>
> **Root cause, confirmed from source, not inferred from the error message alone:**
> `optimum-cli export executorch --task depth-estimation` fails with
> `TypeError: 'NoneType' object is not callable` inside `optimum`'s `main_export`, because
> the task lookup returns `None`. Inspecting the installed `optimum-executorch==1.1.0`
> package directly (`optimum/exporters/executorch/tasks/*.py` and `task_registry.py`)
> shows exactly seven registered export tasks: `automatic-speech-recognition`,
> `image-classification`, `text-generation`, `fill-mask`, `image-text-to-text`,
> `multimodal-text-to-text`, `text2text-generation`. **`depth-estimation` is not among
> them** — there is no `tasks/depth_estimation.py`, and nothing vision-dense-prediction
> shaped exists in this version at all (`image-classification` is the nearest vision task,
> and is structurally the wrong shape — a single-label head, not a per-pixel output).
>
> **The CLI accepts `--task depth-estimation` at all because that name is real — just in
> the wrong registry.** `depth-estimation` → `AutoModelForDepthEstimation` is a genuine,
> documented task in `optimum`'s general-purpose `TasksManager` (the shared task
> vocabulary used across every export backend — ONNX included). But whether a specific
> backend can actually *act* on a task is decided by that backend's own registry, and
> `optimum-executorch`'s is separate, smaller, and does not implement this one. That is
> why the failure surfaces as a `None`-callable crash deep inside `main_export`, rather
> than an upfront "unsupported task" error: the generic layer accepts the name, the
> ExecuTorch-specific layer has nothing registered under it.
>
> **Re-verified 2026-10-19 — still absent, including on unreleased code.** 1.1.0 remains
> the latest PyPI release (confirmed against PyPI's own version listing). `optimum-executorch`'s
> unreleased `main` branch has moved since 1.1.0 — it has since added an
> `object_detection.py` task that did not exist before — but still has no
> `depth_estimation.py` anywhere. This was not a temporary gap this project happened to
> catch mid-release; it is the current and still-advancing state of the library, and depth
> support has not been the thing that advanced.
>
> **This is a library gap, not a model, recipe, or project defect.** The documented
> command in Lesson 04's step-4 prompt cannot succeed against `optimum-executorch` until
> Hugging Face adds a depth-estimation recipe to the ExecuTorch backend specifically (not
> just to the general task vocabulary, which already lists it), or this project builds a
> lower-level export path directly on ExecuTorch's core `torch.export` / `to_edge` /
> `to_executorch` APIs, bypassing `optimum`'s task abstraction entirely — a substantially
> larger undertaking than this step's scope, and not attempted here without explicit
> direction to do so.

#### The lower-level path: attempted 2026-10-19, fixed and shipped 2026-10-19

> **Both problems below are resolved. This is now the real export path** —
> `src/smart_scene_analyzer/export/executorch.py`'s `export_depth()` calls it directly,
> replacing the `optimum-cli` call documented above. Bypassing `optimum` entirely and
> calling ExecuTorch's own core APIs directly (`torch.export.export` → PT2E int8
> quantization → `to_edge_transform_and_lower` with `XnnpackPartitioner` →
> `.to_executorch()`) **produces a real, shippable artifact**: `[1,3,518,518]` float32
> input, native `[1,518,518]` unresized output, no resize inside the graph. This confirms
> the earlier `optimum-cli` failure was specifically a missing *recipe* in that one
> library, not evidence Depth Anything V2 is structurally unexportable to ExecuTorch —
> exactly the distinction ADR 0001's amendment flagged as worth getting right.
>
> **Problem 1 (size) — fixed by quantizing, same idea as detection's `quantize=8`, a
> different toolchain because this path has no CLI flag to do it for you.** ExecuTorch's
> own PT2E flow — `XNNPACKQuantizer` with `get_symmetric_quantization_config(is_per_channel=True)`,
> `prepare_pt2e` → calibrate on 8 real test-split images → `convert_pt2e` — run *before*
> `to_edge_transform_and_lower`, not after. Unquantized float32 was 94.45 MiB (2.4× N9a's
> entire 40 MB combined budget on its own); the quantized artifact is **24.35 MiB**.
> Combined with detection's 2.90 MiB: **27.25 MiB, inside the 40 MB budget**, measured
> rather than estimated, for the first time since ADR 0001 was written.
>
> **Problem 2 (the 0.74 "fidelity gap") — root cause was never `to_edge()`. It was a
> non-contiguous input tensor, misdiagnosed by the earlier triangulation.** Re-running the
> same stage-by-stage comparison (`torch.export` alone / bare `to_edge()` / `to_edge_transform_and_lower`
> with XNNPACK) reproduced the 0.74 correlation initially — but the triangulation table
> itself was a red herring, because all three stages were being driven by the *same*
> corrupted example tensor. `_letterbox_square`'s `arr.transpose(2, 0, 1)[None, ...]`
> inserts the batch axis via NumPy's `newaxis`, which gives that axis **stride 0**
> (confirmed by inspecting the traced graph's placeholder node directly: `pixel_values`
> carried `stride=(0, 1, 1554, 3)`). `torch.from_numpy` preserves that non-contiguous
> layout straight through `torch.export`. Depending on graph shape, this either crashes
> `to_executorch()` outright (`"0 in strides is not supported for ExecuTorch"` — this is
> also where the project's own `_run_optimum_cli_export`-era attempts would have failed
> had they gotten far enough to need real strides) or, as happened here, slips through and
> corrupts the `.pte`'s recorded input layout so the runtime reads the buffer with the
> wrong physical layout: shape-correct, values scrambled. **Fix: `.contiguous()` the
> example tensor before `torch.export.export`** (`np.ascontiguousarray(...)` in
> `_letterbox_square`, defensive `.contiguous()` again in `_run_pte`). Verified end to end
> through the actual ExecuTorch runtime (`executorch.runtime.Runtime`, not just the
> in-process `EdgeProgramManager`, which is pure Python and would not have caught a
> runtime-level layout misread):
>
> | Pipeline | vs. eager, same input, run via `Runtime.get()` |
>|---|---|
> | Before fix (non-contiguous input) | Rank correlation **0.74** |
> | After fix (`.contiguous()`), unquantized | Rank correlation **1.0000**, max abs diff 5.2×10⁻⁵ (float32 rounding) |
> | After fix **and** int8 quantization, shipped artifact, mean of 8 fresh held-out images | Rank correlation **0.9447** (min 0.8748, max 0.9842, std 0.0387) — real quantization noise on a fine-grained regression task, not a lowering bug |
>
> The sign convention (larger=nearer) was never inverted in any of these — always positive
> correlation, confirmed again on the shipped artifact via `_verify_sign_convention`
> (ρ = 0.669 against this project's own `estimate_relative_inverse_depth` reference
> function, which is a cruder heuristic than the HF model itself and not expected to track
> it as tightly as eager-vs-exported does).
>
> **What this means for the op-level root-cause question the earlier write-up left open:**
> it is moot. There was no `to_edge()` decomposition bug to find — the op-by-op diff
> against `EdgeProgramManager` that was previously deferred would have found nothing,
> because every stage downstream of the placeholder was faithfully propagating a stride-0
> input it should never have received. The lesson is about the preprocessing boundary
> (`torch.from_numpy` on a `[None, ...]`-constructed array), not the lowering pipeline.

VERIFIED (export, 2026-10-19 — `export_depth()`, actual `scripts/export.py --depth` run):

| Field | Value |
|---|---|
| Format | ExecuTorch (`.pte`), built via `torch.export` + `to_edge_transform_and_lower`/`XnnpackPartitioner`, not `optimum-cli` |
| Input tensor | `[1, 3, 518, 518]`, float32, **NCHW**, contiguous — fixed by the ViT-S/14 patch size (`518 = 14 × 37`; `640` is not a legal edge length for this backbone) |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`, per channel |
| Output tensor | One tensor, native model resolution, `[1, 518, 518]`, unresized — any resize to `MODEL_INPUT` (640×640) belongs to the depth boundary at inference time, not inside the exported graph |
| Output semantics | `relative-inverse-depth-larger-is-nearer` — no unit, arbitrary per-image scale. Verified against this project's reference depth function, not assumed from the export tool |
| Label order | N/A — depth has no classes |
| Quantization | int8, PT2E flow (`XNNPACKQuantizer`, per-channel symmetric) |
| Calibration set | 8 real test-split images, deterministic (sorted siblings of the probe image) |
| Size | **24.35 MiB** (94.45 MiB unquantized) |
| Fidelity vs. PyTorch eager | Mean rank correlation **0.9447** (min 0.8748, max 0.9842, std 0.0387), n=8 fresh held-out images, measured via the actual ExecuTorch runtime |

**Why two squares and not one.** Both inputs are the same aspect-preserving, centre-padded
letterbox of `SOURCE`, at two edge lengths. The nearest common legal square for both models
(672, since `672 = 14 × 48`) was rejected: 672² is 2,304 patches against 518²'s 1,369 — 68%
more work on depth, the slower model, on every single inference, and it would re-export
detection at a size it was never evaluated at. The two squares relate by one uniform
factor, with no offset term: `scale_depth = scale_det × (518/640)`, same for both pads —
so a depth raster resampled 518²→640² lands in exact `MODEL_INPUT` geometry.

## Training configuration

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Image size | 640 |
| Batch size | 8 |
| Device | `mps` (auto-detected) |
| Optimizer / LR | default (`optimizer=auto`, `lr0=0.01`) |
| Early stopping | default `patience=100` — i.e. not triggered within 50 epochs |
| Wall time | ~100 min (6,004s) through epoch 29 — **interrupted, not finished** |

## Metrics

**This is an interrupted run, not the finished baseline.** `runs/detect/runs/v1-baseline/`
was configured for 50 epochs; `results.csv` stops at epoch 29. `best.pt` and `last.pt` are
identical in size and timestamp, consistent with an interruption rather than
early-stopping on convergence. The numbers below are real, measured on the test split, but
they describe a ~58%-trained model. Full per-class diagnosis, the confusion matrix, and
the single recommended next action (finish training before any data or taxonomy change)
are in `docs/evaluation-v1.md`.

**Measured on:** dataset v1, **test** split, 772 images, 4,995 instances. Confidence
0.001, IoU 0.7 — the full confidence sweep, so the precision-recall curve isn't truncated
(not confidence 0.25 / IoU 0.5 as an earlier draft of this row stated before this run was
evaluated).

| Class | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| bed | 116 | 0.722 | 0.807 | 0.824 | 0.621 |
| bin | 175 | 0.830 | 0.723 | 0.818 | 0.618 |
| chair | 2,352 | 0.773 | 0.764 | 0.820 | 0.576 |
| door | 122 | 0.553 | 0.582 | 0.609 | 0.458 |
| lamp | 148 | 0.761 | 0.559 | 0.616 | 0.406 |
| pillow | 392 | 0.687 | 0.579 | 0.634 | 0.386 |
| screen | 158 | 0.559 | 0.554 | 0.507 | 0.350 |
| sofa | 155 | 0.641 | 0.506 | 0.612 | 0.453 |
| storage | 457 | 0.606 | 0.510 | 0.576 | 0.411 |
| table | 920 | 0.717 | 0.682 | 0.732 | 0.534 |

**Aggregate:** mAP@50 = **0.675**, mAP@50-95 = **0.481**, mean precision = 0.685, mean
recall = 0.627.

Classes with fewer than 30 instances in the test split — rates below would be noise, not
findings: **none.** The smallest is `bed` at 116 test instances.

## Depth

This project's depth channel is **inference only**. It is not part of this trained model,
and it is not fine-tuned.

| Field | Value |
|---|---|
| Depth model | Depth Anything V2 **Small** (`depth-anything/Depth-Anything-V2-Small-hf`), pretrained, not fine-tuned |
| Parameters | 24,785,089 (fp32 checkpoint 94.59 MiB — see `docs/artifact-budget.md`) |
| Output | **Relative inverse depth** — larger = nearer, arbitrary scale, **not metres** |
| Absolute error | **None reported, and none is computable.** The output has no scale, so there is nothing to compare against in any unit |
| **Quantitative evaluation** | **Performed.** N4 pairwise ordering accuracy = **0.9074** |
| N4 target | ≥ 0.90 — **met** |
| Measured on | dataset v1 **test** split, 120 images, 3,121 annotated object pairs, 2,832 correctly ordered |
| Reference | SUN RGB-D's own `depth_bfx` channel, bit-rotation undone, zero readings masked |
| Additional validation | 8 hand-judged scenes (7/8 correct); floor-vs-wall ordering prior on 40 images (38/40, 95%) |

### Why there *is* a depth metric here

An earlier draft of this card would have said no quantitative evaluation was possible,
because Roboflow stores images and annotations but not depth maps. That is true of the
*platform* and false of this *project*: **SUN RGB-D ships a depth channel, and all 8,332
maps are on disk** under `data/SUNRGBD/*/*/*/depth_bfx/`. They were deliberately excluded
from the Roboflow upload — depth would have tripled the storage bill for data the platform
cannot use — but excluding them from the upload is not the same as not having them.

So N4 is measurable, and it has been measured. What remains impossible is anything
involving **scale**: no absolute error, no unit, no cross-image comparison.

The reference and the prediction run in opposite directions, and this is the subtlety that
makes the measurement easy to get backwards: SUN RGB-D's channel is a *range* reading
where larger means further, while Depth Anything V2 returns *inverse* depth where larger
means nearer. A correct model must therefore **disagree in sign on every pair**. Code that
missed this would report near-zero accuracy and look like a model failure rather than a
script bug. `scripts/eval_depth_ordering.py` documents the inversion at the comparison.

### What a downstream consumer must not do

- **Must not** convert these values into a distance, in any unit.
- **Must not** compare values across two images. The scale is per-image and arbitrary.
- **Must not** average them into anything carrying physical units.
- **May** order objects within one image, and shade or rank by the value.

`units_guard.py` enforces the naming half of this in `src/` and `app/`, blocking any write
whose text contains a metric-depth term — including in comments and UI strings, which is
intended: the claim is as wrong in prose as in an identifier.

> **There is no depth-scope ADR.** Earlier drafts of the course template claimed one
> existed; it does not. The authority for depth being inference-only is
> `docs/requirements.md` — N4, and the out-of-scope entry on metric depth — plus
> `.claude/agents/ml-engineer.md`. Either cite those, or write the ADR; do not cite a
> document that is not there.

### Reference depth calibration

Lesson 03, step 7. This is calibration in the metrology sense — checking the model
against a reference standard — not an attempt to derive a conversion from the model's
output to a physical distance. The reference positions below are **the author's own
manual measurements**, taken with a consumer tape measure at the time each scene was
photographed. They are not ground truth from any sensor or dataset, and they carry no
stated unit — the script that consumes them (`scripts/check_reference_depth.py`) uses
only their order and their ratios, never their scale, so no unit enters this project.

Two scenes were built for this check, genuinely different in camera position and object
layout:

| Scene | Objects | Ordering accuracy | Rank correlation |
|---|---|---|---|
| A — desk, cup/mouse/monitor | 3 | 2/2 = 100% (1 pair excluded: two objects fell within the script's 15% minimum-separation threshold) | −0.500 |
| B — desk + chair, wider range | 4 | 4/4 = 100% | −0.800 |

Pooled across both scenes: **6/6 pairs ordered correctly (100%)**, on the pairs that
cleared the minimum-separation threshold.

**Transfer test.** A conversion (relative inverse depth → reference position) was fit on
Scene A, then applied unmodified to Scene B:

```
fit on scene A:  value = 5.670 * (1/position) + -1.882
  scene A error    8.1%   <- fitted here
  scene B error   53.4%
```

The error on Scene B is roughly **6.6× worse** than on the scene the fit was produced
from. This is the expected and intended result: Depth Anything V2 emits relative inverse
depth on an arbitrary **per-image** scale, so a conversion fit on one photograph describes
that photograph and nothing else. The two scenes were shot from different positions and
cover different distance ranges specifically to rule out the alternative explanation —
that similar errors on both scenes would mean only that they were shot from a
coincidentally similar range, not that the conversion generalizes.

**What this does and does not establish.** The model reliably **ranks** objects within one
image by relative distance — the ordering and rank-correlation numbers above are the
evidence for that, and it is the only claim this project makes about depth (N4). It does
**not** establish, and cannot establish, any mapping from these values to a real-world
position. No such conversion is published anywhere in this project, and `units_guard.py`
blocks any write under `src/` or `app/` that names a metric-depth term, specifically so
that a number this calibration shows to be per-image and non-transferable can never be
presented as a measurement.

## Comparison

Only for v2 and later. Not applicable to this card.

## Known failure modes

**Depth, from the validation above:**

- **Boxes that span a depth gradient reduce unreliably.** The median follows whichever
  surface fills most of the box, so a box containing a table edge running toward the camera
  reports "near" even when the labelled object is far. Three of six scenes in the first
  hand-judged round failed for exactly this reason — the labels were wrong, not the model.
  Consumers comparing two boxes should treat small differences as unresolved.
- **Small boxes cropped by the frame edge are the weakest case.** The one remaining failure
  in the ordering fixture (`screen` over `chair`, margin 0.32 against spreads of 2–4 in
  passing scenes) is a chair back clipped at the bottom-left corner.
- **9.3% of annotated object pairs are ordered wrongly** at N4's measurement conditions.
  Ordering is a strong signal, not a guarantee.

**Detection**, from H3's live-camera comparison against this project's COCO-weights
baseline on the same desk (camera, not a dataset split — see `docs/evaluation-v1.md` for
the quantitative test-split numbers, measured on this same **interrupted, 29-of-50-epoch**
checkpoint):

- **Classes outside COCO's vocabulary do appear.** `table` — this project's taxonomy class
  for "any horizontal work or support surface on legs," including a desk — was detected
  where the COCO-pretrained baseline found nothing on the same surface. This is the
  intended benefit of fine-tuning on an indoor taxonomy COCO was never trained toward.
- **A narrower world than COCO also produces new, confident false positives.** The same
  capture hallucinated `pillow` at **0.75 confidence** on a hand resting near a keyboard —
  nothing resembling a pillow was in frame. COCO's baseline made no comparable error on
  this scene. This is the regression direction H3 exists to surface: classes the model now
  *can* express, it can also express wrongly, with no COCO behavior to compare against.
- **Duplicate, overlapping boxes on one object.** The same desk surface produced two
  separate `table` detections (confidence 0.55 and 0.26) rather than one, an imprecision
  distinct from the misclassification above.
- Objects in this capture that are real but outside the K=10 taxonomy entirely —
  `keyboard`, `mouse`, `cup` — were correctly never predicted by the fine-tuned model.
  Their absence is scope, not a failure; COCO's baseline could name them only because
  COCO's own taxonomy happens to include them.

This is one qualitative capture, not a measured rate — a quantitative detection failure
analysis (confusion matrix, per-class precision/recall) is in `docs/evaluation-v1.md`,
with the same interrupted-run caveat attached.

**Fusion**, from Lesson 04 step 3b's live check — the first time detection, depth, and
`fusion.fuse` ran together on one real frame, not a dataset image
(`captures/fusion_live/fusion_live_annotated.png`, 2026-10-15, same interrupted
29-of-50-epoch checkpoint):

- **`screen` misclassified as `table`, confidence 0.30 — DATA, not TAXONOMY.** The box is
  an external monitor shot nearly edge-on (its thin bezel/edge facing the camera, not a
  frontal view of the display). `docs/taxonomy.md` draws `screen` vs. `table` cleanly — a
  powered display vs. a horizontal support surface — so this isn't a boundary ambiguity
  the way `chair`/`sofa` was in the formal evaluation. The likelier cause is **capture-angle
  bias**: `screen` is already this model's single worst class by every metric in
  `docs/evaluation-v1.md` (mAP50 0.507, P=0.559, R=0.554, and one of the smaller classes in
  the taxonomy at ~1,650 source objects), and an oblique, edge-on monitor view is a plausible
  gap in SUN RGB-D's capture distribution, which skews toward frontal shots of furniture.
  MODEL (the interrupted run) is a contributing factor, not the root cause — a class that
  is already this weak on frontal views was never likely to generalize to an unusual angle.
- **The `mixed_region` flag on the `chair` detection (spread=0.300) is RIGHT, not a bug.**
  The box correctly finds a real chair (confidence 0.74) but is loose — it also encloses a
  wooden desk edge, a mug, and a bag at clearly different distances. `depth_spread`'s whole
  purpose is disclosing exactly this: a box whose single relative-depth value is
  summarizing more than one surface. This is the fusion layer doing its job, not failing
  at it — a tighter box would have reported a lower spread, and the layer has no way to
  know the box is loose except by noticing the mixture, which it did.
- **Near-to-far order was plausible but not sharply resolved** (`screen` rank 0 at 3.858,
  `chair` rank 1 at 3.798 — a 1.6% gap). The depth render shows the monitor edge genuinely
  reading slightly brighter/nearer than the chair region, so the order is probably right,
  but the margin is small enough that this pair should not be read as a confident ranking —
  consistent with `docs/model-card-v1.md`'s own depth-calibration finding that small
  differences are unresolved.
