"""Non-maximum suppression. Pure, space-agnostic.

**Single responsibility:** drop duplicate boxes, and nothing else — no decoding, no
coordinate conversion, no thresholding on confidence.

Space-agnostic by design: it takes boxes in whichever space the caller is in and returns
boxes in that same space. On this pipeline it is called in `DET_INPUT`, before the
conversion to `SOURCE`, because suppressing before conversion means one fewer transform
applied to boxes that are about to be discarded.

The IoU threshold and the per-image cap are **assumptions, not requirements** (A7, A8).
`docs/requirements.md` is silent on both, and both move N3 directly — the IoU threshold by
changing what counts as a duplicate, the cap by truncating crowded scenes. They belong in
the artifact manifest so that the app and the reference use the same numbers, rather than
in two independent constants in two languages.
"""

from __future__ import annotations

from collections.abc import Sequence

from smart_scene_analyzer.geometry.boxes import iou as box_iou
from smart_scene_analyzer.types import Detection


def nms(
    detections: Sequence[Detection],
    iou_threshold: float,
    max_detections: int,
    class_agnostic: bool = False,
) -> list[Detection]:
    """Greedy NMS: sort by confidence, keep the best, drop anything that overlaps it past
    `iou_threshold`, repeat. Same space in, same space out. An empty input returns an
    empty list, which is a successful result (F8).
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if len(kept) >= max_detections:
            break
        suppressed = False
        for k in kept:
            if not class_agnostic and k.class_id != candidate.class_id:
                continue
            if box_iou(k.bbox, candidate.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept
