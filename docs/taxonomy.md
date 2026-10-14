# Object taxonomy

**K = 10.** Accepted 2026-09-10. Rationale and rejected alternatives:
[`decisions/0002-object-taxonomy.md`](decisions/0002-object-taxonomy.md).

**This file is an index, not a list.** The number in the first column *is* the integer
class ID in every YOLO label file, every dataset version, the exported artifact's label
array, and C11's `labels` in `architecture.md`. Reordering these rows silently mislabels
every downstream artifact and nothing raises an error — an app built against a different
order draws correct boxes with wrong names.

**The order is alphabetical, and that is not a style choice — it is the export's.**
Roboflow sorts class names alphabetically when it exports a dataset version, and this is
not configurable. Whatever order this file declares, `data.yaml` comes back
`bed, bin, chair, …`. So this file is numbered alphabetically to agree with the artifact
*by construction* rather than by vigilance. See
[`decisions/0003-class-id-order-follows-the-export.md`](decisions/0003-class-id-order-follows-the-export.md).

**"Append only" is therefore not sufficient.** Under alphabetical sorting, adding `desk`
does not append at ID 10 — it *inserts* at index 3 and shifts six classes down. Adding a
class is a retrain and a re-export, not an append.

## Class list

**This heading is load-bearing.** `scripts/verify_export.py` anchors its class-list match
to it; without the heading it scans the whole file, and any other table with
`| <int> | <name> |` rows could override what it reads. Do not rename it.

| ID | Class | Source labels mapped in (exact match) | Objects | Images |
|---|---|---|---|---|
| 0 | `bed` | `bed` | 1,188 | 863 |
| 1 | `bin` | `garbage_bin`, `recycle_bin` | 1,464 | ~1,100 |
| 2 | `chair` | `chair`, `sofa_chair`, `stool`, `bench`, `ottoman` | 25,032 | ~5,300 |
| 3 | `door` | `door` | 1,206 | 1,004 |
| 4 | `lamp` | `lamp` | 1,417 | 925 |
| 5 | `pillow` | `pillow` | 3,564 | 1,009 |
| 6 | `screen` | `monitor`, `tv`, `laptop` | 1,651 | ~950 |
| 7 | `sofa` | `sofa` | 1,371 | 862 |
| 8 | `storage` | `cabinet`, `shelf`, `drawer`, `night_stand`, `bookshelf`, `dresser` | 4,325 | ~2,600 |
| 9 | `table` | `table`, `desk`, `endtable`, `coffee_table` | 9,808 | ~4,400 |

Counts are over the **8,332 locally present SUN RGB-D RGB images** (dataset state of
2026-09-10), totalling 51,026 mapped objects of 65,990 annotated. Image counts for merged
classes are approximate because one image may contain several source labels that merge to
the same class.

## Inclusion rule, per class

Each rule states what the class *is*, so that a borderline object is decided by the rule
rather than by whoever is labelling that day.

| Class | Include | Exclude |
|---|---|---|
| `chair` | Any single-person seat, with or without arms or back: dining chair, office chair, armchair, stool, bench, ottoman/footstool | Multi-person seating (→ `sofa`) |
| `table` | Any horizontal work or support surface on legs, at any height: dining table, desk, side table, coffee table | Fixed built-in surfaces attached to cabinetry (→ omitted; `counter` is not in K) |
| `storage` | An enclosure or set of shelves whose function is holding things: cabinet, cupboard, chest of drawers, bookshelf, open shelving, a single drawer front | Boxes and bags, which are containers but portable (→ omitted) |
| `pillow` | A soft rectangular cushion on a bed or seat | Folded cloth, blankets, mattresses (→ omitted) |
| `screen` | A powered display surface: monitor, television, open laptop | Whiteboards, mirrors, framed pictures (→ omitted); a closed laptop |
| `bin` | An open or lidded waste or recycling receptacle | Boxes, baskets, bags (→ omitted) |
| `lamp` | A discrete light fixture with a visible body: table lamp, floor lamp, sconce | Ceiling-recessed lighting with no body |
| `sofa` | Upholstered seating for two or more, one continuous piece | Armchairs and single seats (→ `chair`) |
| `door` | A hinged or sliding door leaf, open or closed, plus its visible frame | Doorways with no leaf; cabinet doors (→ `storage`) |
| `bed` | A sleeping platform including mattress and frame | The pillows on it, which are labelled separately (→ `pillow`) |

## What is omitted, and what that costs

845 of the 855 source labels are **omitted**: 14,964 objects, 22.7% of the annotations on
these images. They are dropped, not mapped to a background class.

The consequence is real and worth stating plainly: an omitted object is still *in the
image*, just unlabelled. Training therefore teaches the detector to produce no box for a
sink, a counter, a whiteboard, or a box. That is the standard and accepted trade for a
small K, but it means **a missed sink is not a recall failure — it is the taxonomy working
as specified.** `error-triage` should classify it TAXONOMY, not MODEL.

The tail is why K is small: of the 855 source labels, 290 appear exactly once and 627
appear fewer than ten times on these images. A class with nine instances cannot be
learned, split across train/valid/test, or measured.

### 608 uploaded images end up with no label at all

Converting the 8,332 uploaded images against this taxonomy produces **7,724** label
files, not 8,332. The gap is itemized, not lost:

| | Images |
|---|---|
| Converted, with at least one box | 7,724 |
| Never annotated in the source `.mat` | 42 |
| Annotated, but **every** object falls outside K=10 | 566 |

The 566 are real rooms — a bathroom holding only a sink and a toilet, a corridor holding
only a whiteboard. They are not corrupt and not a converter bug.

**They are, however, a decision for version generation.** An image with no label asserts
"this scene contains none of the 10 classes." For the 42 that were never annotated that
assertion is unverified and probably false. Include them and the detector is trained
toward silence on 608 scenes; exclude them and the dataset is 7,724 images. The converter
writes no label file for either group, so they are excluded by default — including them
would require adding them back deliberately.

## Matching is exact, never substring

The converter must compare source labels by **exact string equality**. Substring matching
is wrong here and the dataset contains the trap:

- `dresser_mirror` (169 objects) contains `dresser` but is a **mirror** → omitted
- `sofa_chair` (1,888) contains `sofa` but is a single armchair → `chair`, not `sofa`
- `garbage_bin` and `recycle_bin` both contain `bin`, and both map to `bin` — correct by
  the table, not by the substring

## Adding a class later

**There is no safe append.** Under alphabetical sorting a new class lands wherever its
name falls: `desk` inserts at ID 3 and shifts `door`, `lamp`, `pillow`, `screen`, `sofa`,
`storage` and `table` down one each. Every existing label file, dataset version, and
exported artifact is invalidated, and nothing errors — the app simply draws correct boxes
with wrong names.

Adding or renaming a class is therefore a **retrain and a full re-export**, recorded in a
superseding ADR. See
[`decisions/0003-class-id-order-follows-the-export.md`](decisions/0003-class-id-order-follows-the-export.md).
