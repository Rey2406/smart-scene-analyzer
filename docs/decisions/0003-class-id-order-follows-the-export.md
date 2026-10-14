# ADR 0003 — Class ID order follows the export

- **Status:** accepted
- **Date:** 2026-09-17
- **Deciders:** Project team (Lesson 02, step 12)
- **Amends:** [ADR 0002](0002-object-taxonomy.md) — the class *set* stands unchanged; only
  the *order* is superseded.

## Context

ADR 0002 fixed K = 10 and froze an order, stating it was "frequency-descending on the
2026-09-10 counts by coincidence" and "must not be re-sorted when counts change." The
intent — an order that does not drift with the data — was right. The order chosen was
wrong, and for a reason ADR 0002 did not know:

**Roboflow sorts class names alphabetically when it exports a dataset version, and this
is not configurable.** Whatever order `docs/taxonomy.md` declares, `data.yaml` comes back
`bed, bin, chair, door, lamp, pillow, screen, sofa, storage, table`.

The two orders disagreed on nine classes out of ten:

| Class | ADR 0002 ID | Export ID |
|---|---|---|
| bed | 9 | 0 |
| bin | 5 | 1 |
| chair | 0 | 2 |
| door | 8 | 3 |
| lamp | 6 | 4 |
| pillow | 3 | 5 |
| screen | 4 | 6 |
| **sofa** | **7** | **7** |
| storage | 2 | 8 |
| table | 1 | 9 |

`sofa` agreeing by coincidence is the worst property of this table, not a mitigating one:
a spot-check that happens to land on `sofa` passes, and confirms nothing.

Nothing would have raised an error. YOLO label files store integers; the app's `labels`
array maps integers to names. A mismatch produces an app that detects correctly, draws
boxes correctly, and names every one of them wrong — the exact failure mode
`CLAUDE.md`'s artifact contract describes as "an app built against last week's label order
runs, draws boxes, and names them wrong."

Caught before any dataset version existed, so no immutable artifact carries the wrong
order and no credits were spent on one.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| **A — Keep frequency order; renumber the export afterwards** | Preserves ADR 0002 as written; frequency order is marginally more readable | Requires a remap step between every export and every consumer, forever, in Python *and* TypeScript. Adds a translation layer whose only job is to undo a choice we made. One missed call site is a silent mislabel |
| **B — Keep frequency order; treat the export as wrong** | No document changes | The platform is not negotiable. `CLAUDE.md`: "where the two disagree, the platform wins, and your job is to make the document match rather than the other way round" |
| **C — Renumber the taxonomy alphabetically** | The document agrees with the artifact *by construction*, not by vigilance. No remap layer exists to forget. Free right now | Loses the "most frequent first" readability. Destroys the append-only property (see Consequences) |

## Decision

**Option C.** `docs/taxonomy.md` is numbered alphabetically:

```
0 bed   1 bin    2 chair    3 door     4 lamp
5 pillow 6 screen 7 sofa    8 storage  9 table
```

The class set from ADR 0002 is untouched — same ten classes, same source-label mappings,
same inclusion rules. Only the integers change.

The principle generalizes beyond this file: **where a document and a generated artifact
disagree about a contract, the artifact is the contract.** The document's job is to be
checkable against it, not to assert a competing truth.

## Consequences

**Accepted:**

- **"Append only" is dead.** ADR 0002 stated appending at ID 10 was safe. Under
  alphabetical sorting it is not: adding `desk` inserts at ID 3 and shifts seven classes.
  Adding or renaming a class is a retrain and a full re-export, not an append. This is the
  kind of thing that otherwise gets discovered in Lesson 06.
- Frequency is no longer visible in the ID. Per-class counts stay in `docs/taxonomy.md`,
  and per-class AP must be reported anyway (requirements N3), so nothing is lost but a
  reading convenience.

**Required by this change, and free because no version exists yet:**

- `docs/taxonomy.md` — renumbered.
- `build/yolo-sunrgbd/` — regenerated. The 51,019 local label lines carry ADR 0002's
  integers and must be rewritten. The converter reads its mapping from
  `docs/taxonomy.md`, so this is a re-run, not a code change.
- `docs/requirements.md` — the N3 class list restated in the new order.

**Explicitly not required:**

- **The 7,724 annotations already on Roboflow are correct and untouched.** They were
  uploaded with a labelmap, so the platform stored class *names*, not integers. The
  platform never held ADR 0002's numbering. This is why the defect was cheap: the error
  lived only in local files.

**Reversibility:** none, and that is the point. The order is now a function of the class
names, so it cannot drift by accident — only by adding or renaming a class, which is
itself a superseding-ADR event.
