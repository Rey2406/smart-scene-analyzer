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

The cap also bounds the latency allocations for stages S5 and S6: fusion cost is
`max_detections x MAX_SAMPLES`, which is what makes S6's 15 ms a number rather than a hope.

Planned public surface:

    def nms(
        detections: Sequence[Detection],   # xyxy absolute px, any single space
        iou_threshold: float,
        max_detections: int,
        class_agnostic: bool = False,
    ) -> list[Detection]
        \"\"\"-> the surviving detections, same space as the input, sorted by descending
        confidence and truncated to `max_detections`. An empty input returns an empty
        list, which is a successful result.\"\"\"
"""
