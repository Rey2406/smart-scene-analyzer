"""Stage S6 — the Scene Understanding Layer. **The reference implementation.**

**Single responsibility:** given boxes and a dense depth map, derive one depth score per
object — and say how confident it is in that score.

This module exists twice. `app/src/fusion/index.ts` is a **port** of it, and the port is
what actually runs on the handset. The two are held together by the shared JSON fixtures
in `tests/fixtures/fusion/` (contract C12, requirement N21) and by nothing else. When they
disagree, the disagreement is the finding: fix the port, or record why the reference is
wrong. Never adjust a fixture until both sides pass — that destroys the only signal the
arrangement produces.

**Pure by construction.** Arrays and dataclasses in, a dataclass out. No file reads, no
model loads, no clock, no randomness. That is what makes it testable with a hand-written
8x8 grid, and it is also why it is portable to TypeScript at all — an impure fusion layer
could not be ported.

Units, stated once. Depth here is **relative inverse depth**: larger is nearer, the scale
is arbitrary, and two values are comparable only when they came from the same image. No
field in the output is a physical distance and none may be presented as one.

## The two spaces this module holds at once

Detections arrive in `SOURCE` (contract C7). The depth map arrives in `DEPTH_INPUT`
(contract C8) and is deliberately **not** resampled to `SOURCE`. So every box is mapped
into the depth grid with `letterbox_forward(box, depth_map.transform)` — the **depth**
transform, never the detection one. That is the one conversion in this layer, and it is
the reason this is the module with the shared fixture set.

## The reduction: median over an inset box, sampled on a bounded grid

Per detection:

1. `letterbox_forward` the `SOURCE` box into `DEPTH_INPUT`.
2. Intersect with `depth_map.valid`, so no letterbox padding pixel is ever read. Padding
   is grey canvas the model invented a value for, and it is systematically far.
3. Inset by `INSET_FRACTION` of each side (A5: 0.125, keeping the central 75% x 75%),
   floor-clamped to at least 1 px per side. Box corners are background in almost every
   box; the centre is where the object is.
4. Stride the inset region so at most `MAX_SAMPLES` points are read (A6: 1024). Cost
   becomes independent of box area, which is what makes the S6 latency allocation a number
   instead of a hope.
5. Drop non-finite samples.
6. `relative_depth` = the **median** of the survivors.
7. `depth_spread` = interquartile range of the samples divided by the image's robust
   range, clamped to `[0, 1]`.
8. `relative_depth_normalized` = the median's position within that same robust range,
   clamped to `[0, 1]`.
9. `depth_rank` = index after sorting descending by `relative_depth`; 0 is nearest.

The robust range is the 2nd-to-98th percentile span of the depth map inside `valid`.
Percentiles rather than min and max, because one extreme pixel would otherwise compress
every object in the image into the bottom of the scale.

## What it does when the box contains a background gap

A background gap inside the box is the normal case. A chair's box contains the floor
between its legs; every non-convex object does something similar.

- **The median tolerates it up to 50% contamination, by construction.** That is the whole
  argument for the median over the mean: the mean moves with every background pixel, and
  the median does not move at all until background is the majority.
- **The inset removes the cheapest background** — the corners, which are background even
  in most convex boxes.
- **Above 50% contamination the median follows the background, and it does so silently.**
  A thin, elongated, or heavily occluded object can be a minority of its own bounding box.
  This module does not pretend to fix that; it **reports** it. `depth_spread` is high
  exactly when the sampled region straddles a depth discontinuity, which is the signature
  of a box that is mostly background. A score the system knows is unreliable, reported
  without saying so, is worse than no score.

The honest fix is a segmentation mask, and instance segmentation is out of scope
(`docs/requirements.md`). So the reduction is a box statistic, and the spread is how it
admits what it does not know.

Rejected alternatives, and why: **mean** — every background pixel moves it, and the
corners alone bias it; **max** — one noisy pixel decides the object; **centre pixel** —
free, and wrong for any object whose centre is a gap (chairs, tables, bicycles);
**masked region** — correct and unavailable, it needs segmentation; **median over the
un-inset box** — this is the step-3 fallback, strictly worse on convex objects and no
better on the rest.

## Degenerate cases, each with a defined result

| Case | Result |
|---|---|
| Inset collapses below 1 px | Fall back to the un-inset, clipped box |
| Box lies entirely in the padding band | All depth fields `None`, `depth_sample_count = 0` |
| Every sample non-finite | Same |
| `depth_map is None` (N12) | All depth fields `None` for every detection |
| No detections | Empty result. Successful, not an error (F8) |
| Robust range approximately zero | `relative_depth_normalized = 0.5` for all; no zero-division |

**Non-finite values never leave this layer.** They become `None`, which the caller must
handle — a non-finite value propagated into a rendered overlay draws as *something*, and
usually as "nearest".

`depth_map` is `DepthMap | None` in the signature because N12 makes the degraded,
detection-only path a design constraint rather than error handling. Typing it that way
makes the compiler ask about it instead of a reviewer.

Planned public surface:

    def fuse(
        detections: Sequence[Detection],   # xyxy absolute px, SOURCE space (C7)
        depth_map: DepthMap | None,        # DEPTH_INPUT space, relative inverse depth (C8)
        inset_fraction: float = 0.125,
        max_samples: int = 1024,
    ) -> list[FusedDetection]
        \"\"\"Attach a relative depth score to each detection. Boxes are unchanged and stay
        in SOURCE space. Ranks are assigned across the returned list, 0 = nearest.\"\"\"

    def reduce_box(
        depth_map: DepthMap,
        box_source: BoxXYXY,               # xyxy absolute px, SOURCE space
        inset_fraction: float,
        max_samples: int,
    ) -> BoxDepthStatistic
        \"\"\"The per-box reduction of steps 1-7, for one box. Returns the median, the
        spread, and the sample count; all unit-less, larger median is nearer.\"\"\"

    def robust_range(depth_map: DepthMap) -> tuple[float, float]
        \"\"\"The 2nd and 98th percentiles of the map inside `valid`. Unit-less, and
        meaningful only within this one image.\"\"\"
"""
