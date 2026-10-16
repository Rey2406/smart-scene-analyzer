"""Stage S1 — `SourceImage` to `DetectionInput` (contract C3).

**Single responsibility:** build the detection model's input tensor and the
`SOURCE` -> `DET_INPUT` transform that goes with it, as one inseparable pair.

The tensor and the transform are returned together because they are only correct together.
A tensor without its transform cannot have its boxes brought home; a transform without its
tensor is a scale factor for an image nobody built.

**Layout, corrected against the real export (Lesson 04, step 4, 2026-10-15):** this
docstring previously said `[1, DET_SIZE, DET_SIZE, 3]` NHWC uint8, directly quantized.
The actual exported artifact (`app/assets/models/detector.tflite`, via ultralytics'
`litert_torch` path) is `[1, 3, DET_SIZE, DET_SIZE]` **NCHW float32**, scaled to `[0, 1]`
— no quantization at the input boundary at all; `quantize=8` only affects internal
weights/activations. `docs/model-card-v1.md`'s Detection "Exported artifact" table has the
full reconciliation. The artifact was right; this docstring was wrong, and is now fixed
rather than left to mislead the next reader.

Padding uses `pad_value` (A4, `(114, 114, 114)`), split evenly on both sides of the short
axis, applied in pixel space **before** the `[0, 1]` scale.
"""

from __future__ import annotations

import numpy as np

from smart_scene_analyzer.geometry.letterbox import make_transform
from smart_scene_analyzer.types import DetectionInput, SourceImage


def build(
    image: SourceImage,
    size: int,
    pad_value: tuple[int, int, int] = (114, 114, 114),
) -> DetectionInput:
    """Letterbox and scale to `[1, 3, size, size]` NCHW float32 `[0, 1]`, `DET_INPUT`
    space, with the `SOURCE -> DET_INPUT` `LetterboxTransform` attached.
    """
    from PIL import Image

    h, w = image.pixels.shape[:2]
    transform = make_transform(image.size, size, size)

    img = Image.fromarray(image.pixels, mode="RGB")
    new_w = round(w * transform.scale)
    new_h = round(h * transform.scale)
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), pad_value)
    canvas.paste(resized, (round(transform.pad_x), round(transform.pad_y)))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC, [0, 1]
    tensor = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # -> [1, 3, size, size]

    return DetectionInput(tensor=tensor, transform=transform)
