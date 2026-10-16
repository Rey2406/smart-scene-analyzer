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

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from smart_scene_analyzer.geometry.boxes import inset, intersect
from smart_scene_analyzer.geometry.letterbox import letterbox_forward
from smart_scene_analyzer.types import (
    BoxXYXY,
    DepthMap,
    Detection,
    FusedDetection,
    LetterboxTransform,
)

#: `depth_spread` below this treats the image's dynamic range as degenerate (A6-adjacent):
#: everything reads as mid-scale rather than dividing by ~0.
_ROBUST_RANGE_EPS = 1e-9

#: `depth_spread` at or above this is reported as `"mixed_region"` — the box straddles
#: enough of a real depth discontinuity that the single `relative_depth` value is
#: disclosing, not hiding, its own unreliability. A judgment call, named so it is one
#: place to revisit, not a number buried in a conditional.
MIXED_REGION_SPREAD_THRESHOLD = 0.3

DepthQuality = Literal["ok", "unavailable", "fallback_full_box", "mixed_region"]


@dataclass(frozen=True, slots=True)
class BoxDepthStatistic:
    """The per-box reduction, for one box. Unit-less; larger `median` is nearer."""

    median: float | None
    spread: float | None
    sample_count: int
    fell_back_to_full_box: bool


def robust_range(depth_map: DepthMap) -> tuple[float, float]:
    """The 2nd and 98th percentiles of the map inside `valid`. Unit-less, and meaningful
    only within this one image.

    Percentiles rather than min/max, because one extreme pixel would otherwise compress
    every object in the image into the bottom of the scale.
    """
    x1, y1, x2, y2 = depth_map.valid
    h, w = depth_map.values.shape
    px1, py1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    px2, py2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))

    region = depth_map.values[py1:py2, px1:px2]
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return (0.0, 0.0)
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    return (lo, hi)


def reduce_box(
    depth_map: DepthMap,
    box_source: BoxXYXY,
    inset_fraction: float,
    max_samples: int,
) -> BoxDepthStatistic:
    """The per-box reduction (module docstring, steps 1-7), for one box.

    1. Map `box_source` (`SOURCE`) into `DEPTH_INPUT` with the **depth** transform.
    2. Intersect with `depth_map.valid` — no letterbox padding pixel is ever read.
    3. Inset by `inset_fraction` of each side, falling back to the un-inset box if that
       would collapse it below 1px (`geometry.boxes.inset`'s own floor-clamp).
    4. Stride the region so at most `max_samples` points are read.
    5. Drop non-finite samples.
    6. `median` of the survivors.
    7. `spread` = IQR of the samples / the image's robust range, clamped `[0, 1]`.
    """
    box_depth = letterbox_forward(box_source, depth_map.transform)
    region = intersect(box_depth, depth_map.valid)
    if region is None:
        # Box lies entirely in the padding band.
        return BoxDepthStatistic(
            median=None, spread=None, sample_count=0, fell_back_to_full_box=False
        )

    sampled = inset(region, inset_fraction, min_side_px=1.0)
    # `inset` itself falls back to the un-inset box when insetting would collapse it below
    # 1px (its own floor-clamp) — detected here by comparing in and out, so that fallback
    # is a flag the caller can see rather than an invisible branch inside `inset`.
    fell_back = sampled == region and inset_fraction > 0.0

    h, w = depth_map.values.shape
    sx1, sy1, sx2, sy2 = sampled
    px1, py1 = max(0, int(np.floor(sx1))), max(0, int(np.floor(sy1)))
    px2, py2 = min(w, int(np.ceil(sx2))), min(h, int(np.ceil(sy2)))
    if px2 <= px1 or py2 <= py1:
        return BoxDepthStatistic(
            median=None, spread=None, sample_count=0, fell_back_to_full_box=False
        )

    region_w, region_h = px2 - px1, py2 - py1
    area = region_w * region_h
    stride = max(1, int(np.ceil(np.sqrt(area / max_samples)))) if area > max_samples else 1

    ys = np.arange(py1, py2, stride)
    xs = np.arange(px1, px2, stride)
    samples = depth_map.values[np.ix_(ys, xs)].ravel()
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return BoxDepthStatistic(
            median=None, spread=None, sample_count=0, fell_back_to_full_box=False
        )

    median = float(np.median(finite))

    lo, hi = robust_range(depth_map)
    span = hi - lo
    if span < _ROBUST_RANGE_EPS:
        spread = 0.0
    else:
        q25, q75 = np.percentile(finite, [25, 75])
        spread = float(np.clip((q75 - q25) / span, 0.0, 1.0))

    return BoxDepthStatistic(
        median=median, spread=spread, sample_count=int(finite.size), fell_back_to_full_box=fell_back
    )


def fuse(
    detections: Sequence[Detection],
    depth_map: DepthMap | None,
    inset_fraction: float = 0.125,
    max_samples: int = 1024,
) -> list[FusedDetection]:
    """Attach a relative depth score to each detection. Boxes are unchanged and stay in
    `SOURCE` space. Ranks are assigned across the returned list, `0` = nearest, among
    detections that have a value — a detection with no value gets no rank, not a gap in
    the sequence (N12: `depth_map is None` degrades every detection the same way).
    """
    if not detections:
        return []

    if depth_map is None:
        return [
            FusedDetection(
                detection=d,
                relative_depth=None,
                relative_depth_normalized=None,
                depth_rank=None,
                depth_spread=None,
                depth_sample_count=0,
                depth_quality="unavailable",
            )
            for d in detections
        ]

    stats = [reduce_box(depth_map, d.bbox, inset_fraction, max_samples) for d in detections]
    lo, hi = robust_range(depth_map)
    span = hi - lo

    def normalize(median: float | None) -> float | None:
        if median is None:
            return None
        if span < _ROBUST_RANGE_EPS:
            return 0.5
        return float(np.clip((median - lo) / span, 0.0, 1.0))

    def classify(stat: BoxDepthStatistic) -> DepthQuality:
        if stat.median is None:
            return "unavailable"
        if stat.fell_back_to_full_box:
            return "fallback_full_box"
        if stat.spread is not None and stat.spread >= MIXED_REGION_SPREAD_THRESHOLD:
            return "mixed_region"
        return "ok"

    valid_indices = [i for i, s in enumerate(stats) if s.median is not None]
    valid_indices.sort(key=lambda i: (-stats[i].median, detections[i].class_id))  # type: ignore[operator]
    ranks: list[int | None] = [None] * len(detections)
    for rank, i in enumerate(valid_indices):
        ranks[i] = rank

    return [
        FusedDetection(
            detection=d,
            relative_depth=stats[i].median,
            relative_depth_normalized=normalize(stats[i].median),
            depth_rank=ranks[i],
            depth_spread=stats[i].spread,
            depth_sample_count=stats[i].sample_count,
            depth_quality=classify(stats[i]),
        )
        for i, d in enumerate(detections)
    ]


def fuse_detections_with_depth(
    detections: Sequence[Detection],
    depth_map: np.ndarray | None,
    *,
    image_height: int,
    image_width: int,
) -> list[FusedDetection]:
    """Convenience wrapper over `fuse` for callers that only have depth.py's **raw**
    output — an `HxW` array at `SOURCE` resolution, not a `DepthMap`.

    This bridges two contracts that currently disagree about depth-map resolution:
    `fuse`'s `DepthMap` is specified at `DEPTH_INPUT` resolution with a real letterbox
    transform (`docs/architecture.md` R4) — the eventual on-device shape, once the depth
    export exists (it doesn't yet). `depth.estimate_relative_inverse_depth` (Lesson 03)
    instead resizes its output to `SOURCE` resolution directly. Until both sides agree,
    this wraps the raw array in an **identity** transform (scale=1, no padding) rather
    than silently treating `SOURCE` and `DEPTH_INPUT` as the same space everywhere else
    in the codebase does not.
    """
    if depth_map is None:
        return fuse(detections, None)
    if depth_map.shape != (image_height, image_width):
        raise ValueError(
            f"depth_map shape {depth_map.shape} does not match the stated frame size "
            f"{(image_height, image_width)} — pass the array this frame actually produced, "
            "not a cached one from a different resolution."
        )
    identity = LetterboxTransform(
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        in_w=image_width,
        in_h=image_height,
        out_w=image_width,
        out_h=image_height,
    )
    wrapped = DepthMap(
        values=depth_map,
        transform=identity,
        valid=(0.0, 0.0, float(image_width), float(image_height)),
    )
    return fuse(detections, wrapped)


def sort_near_to_far(fused: Sequence[FusedDetection]) -> list[FusedDetection]:
    """Nearest (`depth_rank` 0) first. A detection with no depth value — no `depth_rank`,
    a degenerate box or a missing depth branch (N12) — sorts last; it was never ranked,
    not ranked worst."""
    return sorted(
        fused, key=lambda d: (d.depth_rank is None, d.depth_rank if d.depth_rank is not None else 0)
    )
