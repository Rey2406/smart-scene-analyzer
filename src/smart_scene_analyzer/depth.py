"""Monocular depth estimation with Depth Anything V2 — inference only.

There is no training here and no Roboflow involvement: the platform does not store depth
ground truth, which is why Lesson 02 scoped depth to inference.

**What this module returns, and the whole reason it is written this way.**

The model produces *relative inverse depth*:

- **Larger values are nearer.** The convention is inverted with respect to how far away a
  surface is, and reading it backwards produces output that looks entirely reasonable —
  every value in range, the map full of structure — while being exactly wrong.
- **The scale is arbitrary.** Two images are not comparable to each other; values are
  meaningful only *within* one image.
- **Not a distance, in any unit.**
- **This project holds no ground truth for depth**, so there is nothing to calibrate
  against and no absolute error can be reported.

The inference call is about fifteen lines. The rest of this module exists so the output
cannot be quietly misread downstream.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# Depth Anything V2 Small is a ViT-S/14: its raster edges are multiples of 14, so the
# model's own output grid is not the source image's grid. Both spaces are named below.
PATCH_MULTIPLE = 14


class DegenerateRegionError(ValueError):
    """A box does not enclose any pixel of the map.

    Raised rather than returning a quiet NaN: a NaN propagates into whatever consumes it
    — a JSON field, a sort key, a comparison that silently evaluates false — and arrives
    somewhere far from the box that caused it. An exception stops at the call site.
    """


class DepthModel(Protocol):
    """The seam a test replaces.

    The default suite runs with no network and no weights on disk, so the loader is an
    interface rather than a direct call into `transformers`.
    """

    def __call__(self, image: Any) -> np.ndarray:
        """Return the model's own raster as HxW float32. Larger is nearer."""


_CACHE: dict[str, DepthModel] = {}


def resolve_device(requested: str | None = None) -> str:
    """Pick cuda / mps / cpu, or honour an explicit override."""
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_default_model(device: str) -> DepthModel:
    """Construct the real model. The only side effect in this module."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)  # type: ignore[no-untyped-call]
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(device).eval()

    def run(image: Any) -> np.ndarray:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            predicted = model(**inputs).predicted_depth
        raster: np.ndarray = predicted.squeeze().float().cpu().numpy().astype(np.float32)
        return raster

    return run


def load_model(device: str | None = None, *, loader: Any = None) -> DepthModel:
    """Load the model once per device and cache it.

    Loading inside the inference function would pay the cost on every call and make the
    function untestable without weights. Pass `loader` to substitute a fake.
    """
    resolved = resolve_device(device)
    if loader is not None:
        substitute: DepthModel = loader(resolved)
        return substitute
    if resolved not in _CACHE:
        _CACHE[resolved] = _build_default_model(resolved)
    return _CACHE[resolved]


def _as_array(image: Any) -> np.ndarray:
    """Accept a PIL Image or an HxWx3 uint8 array; return HxWx3 uint8."""
    if hasattr(image, "convert"):  # PIL.Image
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got shape {arr.shape}")
    return arr.astype(np.uint8, copy=False)


# ------------------------------------------------------------------ the two raster spaces


def estimate_raw_raster(image: Any, *, device: str | None = None, loader: Any = None) -> np.ndarray:
    """Run the model and return **its own raster**, unresized.

    Returns:
        HxW float32 on the MODEL raster — edges are multiples of 14 and generally differ
        from the source image's. Larger is nearer. Arbitrary scale.
        Not a distance, in any unit.

    This is deliberately public. Lesson 04 compares this raw grid against the exported
    artifact's output, so it must be reachable rather than buried inside the resize.
    """
    model = load_model(device, loader=loader)
    raster = np.asarray(model(_as_array(image)), dtype=np.float32)
    if raster.ndim != 2:
        raise ValueError(f"model returned shape {raster.shape}, expected HxW")
    return raster


def resize_raster_to_source(raster: np.ndarray, source_hw: tuple[int, int]) -> np.ndarray:
    """Resample the MODEL raster onto the SOURCE image grid.

    Args:
        raster: HxW float32 on the model raster.
        source_hw: (height, width) of the source image, in pixels.

    Returns:
        HxW float32 on the SOURCE grid.

    Kept as its own named step so the two spaces are visible at the call site. Fusion
    asserts one shape and reduces in one space; silently returning the model's grid would
    make every box lookup wrong by a scale factor and still in range.
    """
    from PIL import Image

    target_h, target_w = source_hw
    if raster.shape == (target_h, target_w):
        return raster.astype(np.float32, copy=False)
    resized = Image.fromarray(raster.astype(np.float32), mode="F").resize(
        (target_w, target_h), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32)


def estimate_relative_inverse_depth(
    image: Any, *, device: str | None = None, loader: Any = None
) -> np.ndarray:
    """Estimate relative inverse depth for one image.

    Args:
        image: PIL Image, or HxWx3 uint8 array.
        device: cuda / mps / cpu. Auto-detected when omitted.
        loader: test seam; replaces the model constructor.

    Returns:
        HxW float32 on the **SOURCE** image grid — the same height and width as `image`.

        **Larger values are nearer.** The scale is arbitrary and is comparable only
        within this one image. **Not a distance, in any unit.** This project holds no
        depth ground truth, so no absolute error for these values can be reported and
        none should be inferred.
    """
    arr = _as_array(image)
    raster = estimate_raw_raster(arr, device=device, loader=loader)
    return resize_raster_to_source(raster, (arr.shape[0], arr.shape[1]))


# ------------------------------------------------------------------- region reduction


def region_relative_depth(depth_map: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Reduce the map inside a box to one value, by the **median**.

    Args:
        depth_map: HxW float32 from `estimate_relative_inverse_depth`, on the SOURCE grid.
        box: (x1, y1, x2, y2), **xyxy in absolute pixels** of that same SOURCE grid,
            per CLAUDE.md's convention.

    Returns:
        The median of the clipped region. Larger is nearer; arbitrary scale;
        not a distance, in any unit.

    Raises:
        DegenerateRegionError: the box encloses no pixel — see the four cases below.

    **Median rather than mean, deliberately.** A box around a chair contains the chair
    *and* whatever is visible around it: floor, wall, the table in front of it. The mean
    blends those, so one occluding surface across a corner drags the value toward itself.
    The median is dominated by whichever surface fills most of the box, which is usually
    the object. Neither is correct in general — median is the more robust default, and
    the reason is written here so the next person can disagree with it knowingly.

    **The four degenerate cases, all defined:**

    1. *Zero-area box* (``x1 >= x2`` or ``y1 >= y2``) — raises. There is no region.
    2. *Partly outside the image* — **not** an error. The box is clipped to the image and
       the surviving pixels are reduced. This is the common case near frame edges.
    3. *Entirely outside the image* — raises. Clipping leaves nothing.
    4. *Empty after clipping* (a sliver that rounds away) — raises, same as case 3.

    Raising rather than returning NaN is the point: a NaN survives arithmetic, sorts
    unpredictably, serializes into JSON as `null` or `NaN`, and surfaces far from the box
    that produced it.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be HxW, got shape {depth_map.shape}")

    height, width = depth_map.shape
    x1, y1, x2, y2 = (float(v) for v in box)

    if x1 >= x2 or y1 >= y2:
        raise DegenerateRegionError(f"zero-area box {box!r}: x1>=x2 or y1>=y2")

    # Clip to the image. int() floors the near edges and ceil the far ones so a box that
    # genuinely overlaps by a fraction of a pixel still keeps that pixel.
    cx1 = max(0, int(np.floor(x1)))
    cy1 = max(0, int(np.floor(y1)))
    cx2 = min(width, int(np.ceil(x2)))
    cy2 = min(height, int(np.ceil(y2)))

    if cx1 >= cx2 or cy1 >= cy2:
        raise DegenerateRegionError(
            f"box {box!r} encloses no pixel of a {width}x{height} map after clipping"
        )

    region = depth_map[cy1:cy2, cx1:cx2]
    if region.size == 0:
        raise DegenerateRegionError(f"box {box!r} clipped to an empty region")

    return float(np.median(region))
