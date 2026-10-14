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

> Filled in by **Lesson 04**, not this lesson. An empty row is a to-do; a guessed row is a
> lie the app will act on.

*(empty — no export exists yet)*

## Training configuration

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Image size | 640 |
| Batch size | 8 |
| Device | `mps` (auto-detected) |
| Optimizer / LR | default (`optimizer=auto`, `lr0=0.01`) |
| Early stopping | default `patience=100` — i.e. not triggered within 50 epochs |
| Wall time | *(pending — run in progress)* |

## Metrics

*(pending — the 50-epoch run has not finished. A 2-epoch smoke test reached mAP@50 0.491
on **valid**, which is a floor and not a result: wrong split, and far from convergence.)*

**Measured on:** dataset v1, **test** split, 772 images, confidence 0.25, IoU 0.5

| Class | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| *(pending)* | | | | | |

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

*(partially pending — detection failure modes need the finished run and predictions
examined on real images.)*

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

**Detection:** *(pending the finished run — see Metrics.)*
