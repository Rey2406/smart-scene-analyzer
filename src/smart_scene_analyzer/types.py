"""The data contracts of `docs/architecture.md` §4, as declarations.

**Single responsibility:** hold every cross-boundary type in one place, with its
coordinate space and its units stated in the field docs — so that a contract cannot be
described in prose in one file and assumed differently in another.

These types are declarations only. Behaviour lives in the modules that produce and consume
them. Nothing in this module imports a model runtime, touches a file, or does arithmetic.

Contract IDs (C1..C13) below refer to `docs/architecture.md` §4. Fields marked
EXPORT-OWNED are fixed by the export step and read from `model_contract.json` (C11); the
values in this project's code are assumptions until that manifest exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

# `xyxy` in absolute pixels of a named space: (x1, y1, x2, y2), x1 <= x2, y1 <= y2.
BoxXYXY = tuple[float, float, float, float]

BoxEncoding = Literal["cxcywh", "xyxy"]
TensorLayout = Literal["NHWC", "NCHW"]

#: A tensor whose dtype is fixed by the export rather than by this package (C11).
ExportOwnedTensor = npt.NDArray[Any]


@dataclass(frozen=True, slots=True)
class Size:
    """A pixel size. Which space it belongs to is stated by the field that holds it."""

    w: int
    h: int


@dataclass(frozen=True, slots=True)
class ImageRef:
    """C1 — a handle to a local image.

    `uri` is a local scheme (`file://`, `ph://`). A remote scheme is a defect: no image
    on this path may be fetched over a network (N1a, F10).
    """

    uri: str
    reported_size: Size | None


@dataclass(frozen=True, slots=True)
class SourceImage:
    """C2 — the decoded working image. Defines the `SOURCE` coordinate space.

    - `pixels`: uint8 `[H_src, W_src, 3]`, RGB, channel-last (HWC). Not BGR, not RGBA.
      Range `[0, 255]`.
    - `size`: `SOURCE` space, absolute pixels, equal to `pixels.shape[1], pixels.shape[0]`.
    - `original_size`: the pixel grid of the file the picker returned, before the
      downscaling decode required by N7.
    - `decode_scale`: `SOURCE` pixels per file pixel, `<= 1.0`.

    EXIF orientation is applied during decode, so `SOURCE` is display-upright and no
    downstream module rotates anything.
    """

    pixels: npt.NDArray[np.uint8]
    size: Size
    original_size: Size
    decode_scale: float


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """C3t — an aspect-preserving letterbox mapping `SOURCE` to one model-input space.

    Forward:  `x_model = x_source * scale + pad_x`
    Inverse:  `x_source = (x_model - pad_x) / scale`

    - `scale`: model-input pixels per `SOURCE` pixel.
    - `pad_x`, `pad_y`: model-input pixels added on the left and top respectively.
    - `in_w`, `in_h`: the `SOURCE` size this transform was computed from.
    - `out_w`, `out_h`: the model-input canvas size.

    **A transform is not interchangeable between models.** Detection and depth have
    different input sizes and therefore different scales and pads. Passing the detection
    transform to a depth lookup returns a number, in range, and wrong.
    """

    scale: float
    pad_x: float
    pad_y: float
    in_w: int
    in_h: int
    out_w: int
    out_h: int


@dataclass(frozen=True, slots=True)
class DetectionInput:
    """C3 — the detection model's input tensor. EXPORT-OWNED.

    - `tensor`: `[1, DET_SIZE, DET_SIZE, 3]`, **NHWC**. dtype, quantization scale and
      zero point, and whether values were scaled to `[0, 1]` before quantization are all
      fixed by the int8 export and recorded in `model_contract.json` (C11).
    - `transform`: `SOURCE` -> `DET_INPUT`.

    `[1, 640, 640, 3]` uint8 NHWC is not interchangeable with `[1, 3, 640, 640]` float32
    NCHW; a silent swap produces plausible garbage rather than an error.
    """

    tensor: ExportOwnedTensor
    transform: LetterboxTransform


@dataclass(frozen=True, slots=True)
class DepthInput:
    """C4 — the depth model's input tensor. EXPORT-OWNED.

    - `tensor`: `[1, 3, DEPTH_SIZE, DEPTH_SIZE]` float32, **NCHW** — a different layout
      from `DetectionInput`, on purpose, because the two runtimes differ.
    - `transform`: `SOURCE` -> `DEPTH_INPUT`. **Not** `DetectionInput.transform`.
    """

    tensor: npt.NDArray[np.float32]
    transform: LetterboxTransform


@dataclass(frozen=True, slots=True)
class RawDetectorOutput:
    """C5 — the detection model's unprocessed output. EXPORT-OWNED.

    Tensor count, order, shape, dtype, box encoding (`cxcywh` or `xyxy`), and whether
    coordinates are normalized to `[0, 1]` or are `DET_INPUT` pixels are all stated in
    `model_contract.json` (C11) and never guessed here.
    """

    tensors: tuple[ExportOwnedTensor, ...]


@dataclass(frozen=True, slots=True)
class Detection:
    """C6 / C7 — one detected object.

    - `bbox`: `xyxy`, absolute pixels. **The space depends on the boundary** and is stated
      by the function that produced the list: `DET_INPUT` at C6 (straight out of NMS,
      possibly overlapping the letterbox padding bands, unclipped), and `SOURCE` at C7
      (after `letterbox_inverse` and clipping — this is the frame F3 requires).
    - `class_id`: index into the ordered `labels` array of C11. The taxonomy is an open
      decision; nothing here computes on the class count.
    - `confidence`: `[0.0, 1.0]`.

    An empty `list[Detection]` is a successful result, not an error (F8).
    """

    bbox: BoxXYXY
    class_id: int
    label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class DepthMap:
    """C8 — the dense depth output (F5), in `DEPTH_INPUT` space.

    - `values`: float32 `[DEPTH_SIZE, DEPTH_SIZE]`. No batch axis, no channel axis.
    - `transform`: `SOURCE` -> `DEPTH_INPUT`, reused from `DepthInput`.
    - `valid`: the un-padded content rectangle, `xyxy` in `DEPTH_INPUT` pixels. Values
      outside it are letterbox padding — grey canvas the model invented a depth for, and
      systematically far. **Padding is never sampled.**

    **Units: none.** `values` is relative inverse depth: larger is nearer, the scale is
    arbitrary, and values are comparable only within this one image. There is no
    conversion to a physical distance, and this system does not have one to offer.

    The map is deliberately **not** resampled to `SOURCE` resolution: fusion maps each box
    into this grid instead (`docs/architecture.md` R4).
    """

    values: npt.NDArray[np.float32]
    transform: LetterboxTransform
    valid: BoxXYXY


@dataclass(frozen=True, slots=True)
class FusedDetection:
    """C9 — one detected object with its depth score attached.

    - `bbox`: `xyxy`, absolute pixels, **`SOURCE` space** (F3).
    - `relative_depth`: the median of the sampled region. Unit-less relative inverse
      depth, larger is nearer, comparable only within this image. `None` when no depth was
      derivable — the depth branch was unavailable (N12) or the region yielded no valid
      samples. **Never NaN.**
    - `relative_depth_normalized`: `relative_depth` positioned in this image's robust
      (2nd-98th percentile) range, clamped to `[0, 1]`. `1.0` is the nearest end.
      Meaningless across images.
    - `depth_rank`: `0` is the nearest object in this image. Ties break by descending
      `relative_depth`, then by `class_id`.
    - `depth_spread`: `[0, 1]`. The interquartile range of the sampled region divided by
      the image's robust range. High means the box straddles a depth discontinuity, which
      is the signature of a box that is mostly background — the one case the median fails
      at silently.
    - `depth_sample_count`: how many valid samples produced the values above. `0` means
      every depth field is `None`.
    """

    detection: Detection
    relative_depth: float | None
    relative_depth_normalized: float | None
    depth_rank: int | None
    depth_spread: float | None
    depth_sample_count: int


@dataclass(frozen=True, slots=True)
class DepthPreview:
    """C9 — the shaded depth overlay's source (F9).

    - `shade`: uint8 `[DEPTH_SIZE, DEPTH_SIZE]`, `0..255`, `DEPTH_INPUT` space.

    For shading and ordering only. It is a rank-preserving remap of `DepthMap.values`, not
    a measurement, and the UI may not label it with a distance.
    """

    shade: npt.NDArray[np.uint8]
    transform: LetterboxTransform


@dataclass(frozen=True, slots=True)
class SceneResult:
    """C9 — the one structure the overlay renders.

    - `detections`: boxes in `SOURCE` space with their depth fields.
    - `depth_available`: `False` means the depth branch failed to load or run; every
      `relative_depth` is then `None` and the result is detection-only (N12). Detection
      failure is not represented here — it fails visibly instead.
    - `source_size`: `SOURCE` space, needed for the `SOURCE` -> `SCREEN` conversion (C10).
    - `detector_version`, `depth_version`: registry versions from C11 (F7).

    `detections == []` with `depth_available is True` is a successful empty result, and
    the UI must render it distinguishably from "loading" and from "the model failed to
    load" (F8).
    """

    detections: tuple[FusedDetection, ...]
    depth_available: bool
    source_size: Size
    detector_version: str
    depth_version: str | None
    depth_preview: DepthPreview | None


@dataclass(frozen=True, slots=True)
class Viewport:
    """C10 — the mapping from `SOURCE` pixels to `SCREEN` points.

    - `view_w`, `view_h`: the image view's size in **density-independent points**, not
      physical pixels.
    - `image_w`, `image_h`: `SOURCE` pixels.
    - `fit`: `"contain"`.

    Contain-fit introduces a third scale and a third pair of offsets, distinct from both
    letterbox transforms. It changes on rotation and on split-view, so it is recomputed
    per layout pass and never cached across one.
    """

    view_w: float
    view_h: float
    image_w: int
    image_h: int
    fit: Literal["contain"]
