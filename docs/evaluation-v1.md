# Evaluation — `v1-baseline`

## Measurement conditions

| Field | Value |
|---|---|
| Model | YOLO11n |
| Weights | `runs/detect/runs/v1-baseline/weights/best.pt` — **not** `runs/v1-baseline/...` as named in the lesson prompt; the actual run nested one level deeper under `runs/detect/` |
| Dataset version | `data/v1` (`scripts/make_local_version.py`, seed=1 — a local split, **not** a Roboflow dataset version) |
| Split | **test** (`data/v1/test`), not valid |
| Images | 772 |
| Instances | 4,995 |
| Confidence threshold | **0.001** for mAP and per-class P/R (the full confidence sweep, so the precision-recall curve isn't truncated); **0.25** separately for the confusion matrix (the conventional operating point — at 0.001 the matrix is dominated by near-zero-confidence noise and the real class-confusion signal disappears under it) |
| IoU threshold | 0.7 (NMS) for both |

**This checkpoint is not the finished baseline.** `runs/detect/runs/v1-baseline/` was
configured for 50 epochs; `results.csv` has 29 rows. `best.pt` and `last.pt` are identical
in size and timestamp, consistent with an interrupted run rather than early-stopping on
convergence. Every number below is for a ~58%-trained model. Per explicit instruction,
this checkpoint is evaluated as-is rather than waiting for a completed run.

**Two cross-references Round 2 normally requires are unavailable.** `docs/dataset-card-v1.md`
does not exist in this project yet, and no Lesson 02 Auto Label audit write-up exists
(only a raw `data/audit-gt/` directory, no doc). Both checks below are marked
**unavailable** rather than skipped silently.

## Per-class results

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| bed | 87 | 116 | 0.722 | 0.807 | 0.824 | 0.621 |
| bin | 124 | 175 | 0.830 | 0.723 | 0.818 | 0.618 |
| chair | 544 | 2,352 | 0.773 | 0.764 | 0.820 | 0.576 |
| door | 101 | 122 | 0.553 | 0.582 | 0.609 | 0.458 |
| lamp | 90 | 148 | 0.761 | 0.559 | 0.616 | 0.406 |
| pillow | 102 | 392 | 0.687 | 0.579 | 0.634 | 0.386 |
| screen | 99 | 158 | 0.559 | 0.554 | 0.507 | 0.350 |
| sofa | 84 | 155 | 0.641 | 0.506 | 0.612 | 0.453 |
| storage | 259 | 457 | 0.606 | 0.510 | 0.576 | 0.411 |
| table | 471 | 920 | 0.717 | 0.682 | 0.732 | 0.534 |

**Aggregate:** mAP50 = **0.675**, mAP50-95 = **0.481**, mean precision = 0.685, mean
recall = 0.627.

No class has fewer than 30 instances in the test split (lowest is `bed` at 116), so none
is flagged as noise.

## Confusion matrix

At conf=0.25, the dominant error by volume in every class is background — the model
missing an instance (predicted background, true class) or hallucinating one (predicted
class, true background), led by `chair` (787 spurious, 397 missed) and `table` (302
spurious, 228 missed). That is expected given `chair` and `table` are this taxonomy's two
largest merged classes (25,032 and 9,808 source objects respectively, per
`docs/taxonomy.md`) and carry the most instances in this split.

**The top five genuine class-vs-class confusions** (excluding background, pooling each
pair's two directions):

1. **`chair` ↔ `sofa`** — 54 (predicted chair/true sofa: 33; predicted sofa/true chair: 21)
2. **`chair` ↔ `table`** — 42 (21 each direction)
3. **`table` ↔ `storage`** — 35 (18 + 17)
4. **`chair` ↔ `bin`** — 8
5. **`sofa` ↔ `bed`** — 6

Pairs 1–3 account for the overwhelming majority of non-background confusion; everything
past rank 5 is single digits.

## Diagnosis

**No individual class crosses the prompt's trigger** (recall below 0.5 or precision below
0.5) — the lowest recall is `sofa` at 0.506, the lowest precision is `door` at 0.553, both
just inside the bound. Rather than force a diagnosis onto a class that technically clears
the bar, the three confusion pairs that dominate the actual error volume are diagnosed
instead, since they are where the evidence concentrates.

### `chair` ↔ `sofa` — **TAXONOMY**, with a DATA aggravating factor

`docs/taxonomy.md`'s inclusion rule draws this boundary on seating *capacity*: `chair` is
"any single-person seat... armchair, stool, bench, ottoman"; `sofa` is "upholstered
seating for **two or more**, one continuous piece." The taxonomy itself flags the trap —
`sofa_chair` (1,888 source objects) contains the substring "sofa" but is a single-person
armchair and must map to `chair`, not `sofa`. Whether a piece of furniture seats one or
two is frequently not recoverable from a single camera angle or a partially-occluded crop,
which makes this a boundary that depends on information the image doesn't always carry —
the definition of a TAXONOMY finding, not a model-capacity one.

The 15:1 instance imbalance between `chair` (2,352) and `sofa` (155) in this split is a
DATA factor that compounds it: on a genuinely ambiguous case, a model trained on 15x more
chair examples has every incentive to default to `chair`. The matrix shows some of this
asymmetry — 33 instances of a true `sofa` predicted as `chair`, against 21 the other
direction — tilted toward the majority class, though not a one-sided collapse.

**Evidence used:** `docs/taxonomy.md` inclusion rule for `chair`/`sofa`; per-class instance
counts from this evaluation; `docs/dataset-card-v1.md` cross-reference — **unavailable**;
Lesson 02 Auto Label audit cross-reference — **unavailable**.

### `table` ↔ `storage` — **TAXONOMY**

`storage`'s inclusion rule explicitly folds in `night_stand` alongside cabinets, chests,
and bookshelves, while `table`'s rule includes "side table" and "coffee table." A
nightstand or a low shelving unit at table height is visually a horizontal surface on
legs unless its enclosed/shelved interior is visible — which depends on viewing angle, not
on anything the object category guarantees. The taxonomy groups these by function
(does it enclose things, or just support them) rather than by silhouette, and the two
functions can look identical from the wrong angle. This is a class-boundary problem, not
an undertrained model.

**Evidence used:** `docs/taxonomy.md` inclusion rules for `table` and `storage`. Dataset-card
and Lesson 02 audit cross-references — **unavailable**.

### `chair` ↔ `table` — **MODEL** (tentative, given the interrupted run)

Unlike the two pairs above, `chair` and `table` have no shared or ambiguous definition in
`docs/taxonomy.md` — a seat and a horizontal support surface are visually distinct
categories. There's no taxonomy line that explains this confusion, and the imbalance
(2,352 vs 920, 2.5:1) is far less extreme than the chair/sofa case. The more plausible
explanation is that this is ordinary localization/classification error in cluttered scenes
from a model that stopped training at 29 of 50 configured epochs. This is the one
diagnosis in this report that rests on the absence of a better explanation rather than on
a specific piece of evidence, which is exactly the kind of claim that needs checking
against a completed run before it's trusted.

**Evidence used:** absence of a taxonomy explanation; the interrupted-training measurement
condition noted above. Dataset-card and Lesson 02 audit cross-references — **unavailable**.

## Recommended action

**Resume or restart `v1-baseline` training to the full configured 50 epochs before making
any taxonomy or data changes.** This checkpoint has only seen 58% of its planned training,
and one of the three dominant confusion pairs (`chair`↔`table`) has no taxonomy
explanation and is tentatively attributed to that alone.

**What I expect to change:** the `chair`↔`table` confusion and the generally mediocre
numbers on `door`, `screen`, `sofa`, and `storage` (all clustered at 0.50–0.65 P/R) should
improve measurably once training completes — if they don't, that would be evidence against
the MODEL attribution above and worth re-opening. **What I expect to persist:** the
`chair`↔`sofa` and `table`↔`storage` confusions should **not** meaningfully close with
more training, because their cause (an ambiguous class boundary) is not something more
gradient steps on the same labels can fix. If a completed run still shows those two
confusions near their current volume, that is confirmation the TAXONOMY diagnosis was
right, not a sign the retrain failed.
