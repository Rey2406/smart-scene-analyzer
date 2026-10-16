"""Operations on `xyxy` boxes, within a single coordinate space.

**Single responsibility:** box arithmetic that does not change coordinate space. Anything
that crosses a space belongs in `letterbox.py`, not here.

Every function takes and returns `xyxy` in absolute pixels, and **both the argument and
the result are in the same space** — whichever space the caller was already in. That is
the invariant that makes this module safe to use from both `SOURCE` and model-input code.
"""

from __future__ import annotations

import math

from smart_scene_analyzer.types import BoxXYXY, Size


def clip(box: BoxXYXY, size: Size) -> BoxXYXY:
    """Clamp to `[0, w] x [0, h]`. May return a zero-area box; the caller decides whether
    to drop it (contract C7 drops it)."""
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), float(size.w)),
        min(max(y1, 0.0), float(size.h)),
        min(max(x2, 0.0), float(size.w)),
        min(max(y2, 0.0), float(size.h)),
    )


def inset(box: BoxXYXY, fraction: float, min_side_px: float = 1.0) -> BoxXYXY:
    """Shrink by `fraction` of each side, keeping the centre. Floor-clamped so each side
    keeps at least `min_side_px`. Used by the fusion reduction (A5): box corners are
    background in almost every box, and the centre is where the object is."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * fraction, h * fraction

    new_w, new_h = w - 2 * dx, h - 2 * dy
    if new_w < min_side_px or new_h < min_side_px:
        return box

    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def area(box: BoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersect(a: BoxXYXY, b: BoxXYXY) -> BoxXYXY | None:
    """`None` when the boxes do not overlap — not a zero-area box, which would be
    indistinguishable from a legitimate degenerate one."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return (ix1, iy1, ix2, iy2)


def iou(a: BoxXYXY, b: BoxXYXY) -> float:
    inter = intersect(a, b)
    if inter is None:
        return 0.0
    inter_area = area(inter)
    union = area(a) + area(b) - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def is_valid(box: BoxXYXY) -> bool:
    """`x1 <= x2` and `y1 <= y2` and no NaN. The invariant every `Detection` asserts."""
    x1, y1, x2, y2 = box
    if any(math.isnan(v) for v in box):
        return False
    return x1 <= x2 and y1 <= y2
