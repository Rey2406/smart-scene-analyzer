"""Stage S5 — raw detector tensors to `Detection` lists (contracts C5 -> C6 -> C7).

**Single responsibility:** interpret the detector's output encoding, and own the single
place where a box crosses from `DET_INPUT` into `SOURCE`.

Pure: arrays in, dataclasses out. No weights, no files, no clock.

Three steps, in this order, and the order matters:

1. **Decode** the raw tensors into candidate boxes in `DET_INPUT` pixels. The encoding
   (`cxcywh` vs `xyxy`, normalized `[0, 1]` vs `DET_INPUT` pixels) comes from the artifact
   manifest — Ultralytics emits both depending on the export path, and guessing costs an
   afternoon. Boxes here may lie in the letterbox padding bands and are not clipped.
2. **Suppress** — `nms.py`, still in `DET_INPUT`. Yields contract C6.
3. **Bring home** — `letterbox_inverse` with the **detection** transform, then clip to
   `[0, W_src] x [0, H_src]`. Yields contract C7, which is the frame F3 requires and the
   frame everything downstream may assume.

**Clipping happens here and only here.** A box that clips to zero area is dropped rather
than kept degenerate: it lay entirely in the padding band, which means it was never in the
image. Clipping in two places means two sets of edge-case behaviour that agree until they
do not.

An empty result is a successful result (F8), at every one of the three steps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from smart_scene_analyzer.export.contract import DetectorOutputSpec
from smart_scene_analyzer.geometry.boxes import clip, is_valid
from smart_scene_analyzer.geometry.letterbox import letterbox_inverse
from smart_scene_analyzer.types import Detection, LetterboxTransform, RawDetectorOutput, Size


def decode_raw(
    raw: RawDetectorOutput,
    spec: DetectorOutputSpec,
    labels: Sequence[str],
    conf_threshold: float,
    det_input_size: int,
) -> list[Detection]:
    """-> boxes in xyxy absolute pixels, `DET_INPUT` space, unclipped (C5 -> pre-NMS).

    This project's real export emits one tensor, `[1, 4 + num_classes, num_boxes]`
    (`cxcywh`, normalized `[0, 1]`) — `spec` states that explicitly rather than this
    function assuming it, so a future export with a different encoding or tensor count
    fails loudly here instead of silently decoding garbage.

    `det_input_size` is the model **input** canvas edge (e.g. 640) that normalized
    coordinates are relative to. It cannot be read from `spec` — the output tensor's own
    shape (`[1, 14, 8400]`: channels x anchors) carries no spatial dimension at all, so
    this must come from the input side, not be guessed from the output's shape.
    """
    tensor = raw.tensors[0]
    arr = np.asarray(tensor, dtype=np.float32)[0]  # drop batch
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # -> [num_boxes, channels]

    box_cols = arr[:, :4]
    class_scores = arr[:, 4:]
    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    keep = confidences >= conf_threshold

    detections: list[Detection] = []
    for box, conf, cls_id in zip(box_cols[keep], confidences[keep], class_ids[keep], strict=True):
        if spec.encoding == "cxcywh":
            cx, cy, w, h = box
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            x1, y1, x2, y2 = box

        if spec.normalized:
            x1, x2 = x1 * det_input_size, x2 * det_input_size
            y1, y2 = y1 * det_input_size, y2 * det_input_size

        detections.append(
            Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                class_id=int(cls_id),
                label=labels[int(cls_id)],
                confidence=float(conf),
            )
        )
    return detections


def to_source_frame(
    detections: Sequence[Detection],
    transform: LetterboxTransform,
    source: Size,
) -> list[Detection]:
    """-> boxes in xyxy absolute pixels, `SOURCE` space, clipped to the image, with
    zero-area results dropped (C6 -> C7)."""
    out: list[Detection] = []
    for d in detections:
        source_box = letterbox_inverse(d.bbox, transform)
        clipped = clip(source_box, source)
        if not is_valid(clipped):
            continue
        x1, y1, x2, y2 = clipped
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append(
            Detection(bbox=clipped, class_id=d.class_id, label=d.label, confidence=d.confidence)
        )
    return out
