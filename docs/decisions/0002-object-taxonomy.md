# ADR 0002 — Object taxonomy: 10 merged indoor classes

- **Status:** accepted
- **Date:** 2026-09-10
- **Deciders:** Project owner (accepted the `dataset-engineer` proposal, Lesson 02)

## Context

`docs/requirements.md` left the class list open and recorded the consequence: N3's
mAP@50 ≥ 0.50 is provisional, because "an mAP target over an unknown number of classes is
not a target." `docs/roadmap.md` names this the highest-priority open decision of Lesson
02 and requires it settled **before** a Roboflow dataset version is generated, since a
version generated against a different taxonomy must be regenerated — which bills again
and invalidates every count in the dataset card.

The source data is SUN RGB-D's 2D bounding-box metadata
(`data/SUNRGBD/SUNRGBDMeta2DBB_v2.mat`), inspected on 2026-09-10 rather than read from
documentation. What the file actually contains:

- 10,335 records; **8,332** have their RGB image present locally, carrying **65,990**
  annotated objects. Zero local images lack a record.
- Boxes are `gtBb2D`, four values, **`xywh` in absolute pixels, origin top-left**. Proven
  by counter-example, not assumed: read as `xyxy`, 62,422 of 80,220 objects (77.8%) become
  geometrically invalid; read as `xywh`, none do.
- **855 distinct class-name strings** on those images, with a severe tail — 290 appear
  exactly once, 627 fewer than ten times.

Two facts about that label set forced the decision:

1. **The tail is unlearnable.** A label with nine instances cannot survive a
   train/valid/test split, cannot be trained, and cannot be measured. Any taxonomy that
   admits it is reporting a per-class AP computed on noise.
2. **Several *frequent* labels are distinctions the annotators did not draw
   consistently.** `table` (5,859), `desk` (2,957), `endtable` (650) and `coffee_table`
   (342) are the same physical object in different rooms; so are `sofa` / `sofa_chair` /
   `ottoman`, and `cabinet` / `shelf` / `bookshelf` / `dresser` / `drawer`. A detector
   asked to separate them learns the room rather than the object.

The second point is the one that matters, because it is invisible in a coverage
percentage. Keeping those labels distinct raises nominal coverage and lowers achievable
mAP, and the resulting confusion-matrix bleed looks like a model problem while being a
specification problem.

Additional constraint from the target platform: this taxonomy is consumed by YOLO11
exported to **int8 TFLite running on a handset** with a 400 ms p95 budget (N1) alongside a
second model. K sets the detection head's width and the postprocessing cost, and int8
quantization degrades a crowded, imbalanced class space more than a sparse one.

## Options considered

| Option | K | Mapped coverage | Min class | Max/min | Pros | Cons |
|---|---|---|---|---|---|---|
| **A — Top 30 source labels, verbatim** | 30 | 82.5% | 309 | 72:1 | No mapping judgment; faithful to the source; highest nominal coverage per unit of effort | Preserves every annotator-inconsistent pair (`table`/`desk`/`endtable`/`coffee_table`, `sofa`/`sofa_chair`/`ottoman`); admits `paper` (444) and `cpu` (362), which a phone camera rarely resolves; widest head and worst int8 behaviour |
| **B — 15 classes, partial merge** | 15 | ~84% | 260 | 96:1 | Best semantic detail for a scene-understanding demo; keeps `sink`, `toilet`, `counter` as distinct affordances | `toilet` at 260 and `sink` at 533 instances give visibly weak per-class AP; imbalance widens to 96:1, which int8 handles poorly; more classes *and* more imbalance than A at the same coverage |
| **C — 10 merged classes** | 10 | 77.3% | 1,188 | 21:1 | Every class exceeds 1,000 instances; tightest imbalance of the three; smallest head and cheapest postprocessing; merges exactly the pairs annotators confused | Lowest nominal coverage; loses `sink`, `counter`, `toilet`, `box`, `whiteboard` entirely; `storage` and `chair` are broad enough that a user could reasonably expect finer labels |

## Decision

**Option C. K = 10, in the frozen order recorded in `docs/taxonomy.md`:**

```
0 chair   1 table   2 storage   3 pillow   4 screen
5 bin     6 lamp    7 sofa      8 door     9 bed
```

51,026 of 65,990 objects map; 14,964 (22.7%) are omitted. Merges collapse precisely the
label groups identified above as inconsistently drawn.

Coverage was rejected as the deciding metric. A is nominally 5.2 points better covered
than C while being the option most likely to miss N3, because its extra coverage is
concentrated in classes that compete with each other. C trades 5 points of nominal
coverage for a minimum class size 3.8× larger and an imbalance ratio 3.4× tighter.

The order is frozen independently of frequency. It is frequency-descending on the
2026-09-10 counts by coincidence; it must not be re-sorted when counts change.

## Consequences

**Accepted:**

- N3's mAP@50 ≥ 0.50 is now a statement over 10 named classes and can stop being
  provisional. `docs/requirements.md` must be updated accordingly.
- 22.7% of annotations are discarded. Omitted objects remain visible in the images, so
  training actively teaches the detector to emit no box for a sink, counter, whiteboard,
  or box. **A missed sink is the taxonomy working, not a recall regression** —
  `error-triage` must classify it TAXONOMY, not MODEL.
- `chair` carries 25,032 objects against `bed`'s 1,188, a 21:1 imbalance. Per-class AP
  must be reported, never a bare mAP; the headline number is dominated by `chair`.
- Class ID order is now a load-bearing contract shared by the label files, the dataset
  version, the exported artifact, and the app's `labels` array. Nothing validates it at
  runtime.

**Rejected consequences we are choosing to live with:**

- No `sink`, `counter`, `toilet`, `box`, or `whiteboard`. If a later lesson needs a
  bathroom or kitchen affordance, that is an appended class at ID 10 and a superseding
  ADR — not an edit to this one.
- `storage` merges an open bookshelf with a closed cabinet. They differ visually more than
  the other merges in this taxonomy, and this is the merge most likely to be revisited.

**Reversibility:** appending at ID 10 is safe. Inserting, renaming, or reordering
invalidates every label file, dataset version, and exported artifact, and requires a
superseding ADR plus a full re-export.
