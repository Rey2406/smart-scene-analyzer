"""Stage S5 — raw detector tensors to `Detection` lists (contracts C5 -> C6 -> C7).

**Single responsibility:** interpret the detector's output encoding, and own the single
place where a box crosses from `DET_INPUT` into `SOURCE`.

Pure: arrays in, dataclasses out. No weights, no files, no clock. Always tested.

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

This function is mirrored in TypeScript in the app, which is where it actually runs
(assumption A15 — if a future export bakes NMS into the graph, contract C5 changes shape
and this module shrinks to step 3).

Planned public surface:

    def decode_raw(
        raw: RawDetectorOutput,
        spec: DetectorOutputSpec,        # encoding + normalization, from the manifest
        labels: Sequence[str],           # ORDERED; index == class_id
        conf_threshold: float,
    ) -> list[Detection]
        \"\"\"-> boxes in xyxy absolute pixels, DET_INPUT space, unclipped (C5 -> pre-NMS).\"\"\"

    def to_source_frame(
        detections: Sequence[Detection],
        transform: LetterboxTransform,   # the DETECTION transform, never the depth one
        source: Size,
    ) -> list[Detection]
        \"\"\"-> boxes in xyxy absolute pixels, SOURCE space, clipped to the image, with
        zero-area results dropped (C6 -> C7).\"\"\"
"""
