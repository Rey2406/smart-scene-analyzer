"""Aspect-preserving letterbox: `SOURCE` <-> a model-input space, both directions.

**Single responsibility:** compute the one transform that relates `SOURCE` to a model
input canvas, and apply it forward to boxes and inverse to boxes.

There are **two** transforms on this path, one per model, because the two models have
different input resolutions. Functions here take the transform as an argument and never
reach for a default; a function that guessed which model it was serving is the bug this
module is arranged to prevent.

Arithmetic (`docs/architecture.md` C3t):

    forward:  x_model  = x_source * scale + pad_x
              y_model  = y_source * scale + pad_y
    inverse:  x_source = (x_model - pad_x) / scale
              y_source = (y_model - pad_y) / scale

`scale = min(out_w / in_w, out_h / in_h)`; padding is split evenly between the two sides
of the short axis (assumption A4 — an export that pads one-sided changes this arithmetic).

Both box functions return **unclipped** boxes. Clipping to the image is a separate,
deliberate step that happens once, at contract boundary C7, in `inference.decode`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from smart_scene_analyzer.types import BoxXYXY, LetterboxTransform, Size


def make_transform(source: Size, out_w: int, out_h: int) -> LetterboxTransform:
    """Compute the SOURCE -> model-input transform for one model's canvas."""
    scale = min(out_w / source.w, out_h / source.h)
    new_w = source.w * scale
    new_h = source.h * scale
    pad_x = (out_w - new_w) / 2
    pad_y = (out_h - new_h) / 2
    return LetterboxTransform(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        in_w=source.w,
        in_h=source.h,
        out_w=out_w,
        out_h=out_h,
    )


def letterbox_image(
    pixels: npt.NDArray[np.uint8],
    transform: LetterboxTransform,
    pad_value: tuple[int, int, int],
) -> npt.NDArray[np.uint8]:
    """`[H_src, W_src, 3]` RGB, SOURCE space -> `[out_h, out_w, 3]` RGB, model-input space."""
    from PIL import Image

    img = Image.fromarray(pixels, mode="RGB")
    new_w = round(transform.in_w * transform.scale)
    new_h = round(transform.in_h * transform.scale)
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (transform.out_w, transform.out_h), pad_value)
    canvas.paste(resized, (round(transform.pad_x), round(transform.pad_y)))
    return np.asarray(canvas, dtype=np.uint8)


def letterbox_forward(box: BoxXYXY, transform: LetterboxTransform) -> BoxXYXY:
    """xyxy absolute px, SOURCE -> xyxy absolute px, model-input. Unclipped."""
    x1, y1, x2, y2 = box
    return (
        x1 * transform.scale + transform.pad_x,
        y1 * transform.scale + transform.pad_y,
        x2 * transform.scale + transform.pad_x,
        y2 * transform.scale + transform.pad_y,
    )


def letterbox_inverse(box: BoxXYXY, transform: LetterboxTransform) -> BoxXYXY:
    """xyxy absolute px, model-input -> xyxy absolute px, SOURCE. Unclipped."""
    x1, y1, x2, y2 = box
    return (
        (x1 - transform.pad_x) / transform.scale,
        (y1 - transform.pad_y) / transform.scale,
        (x2 - transform.pad_x) / transform.scale,
        (y2 - transform.pad_y) / transform.scale,
    )


def content_rect(transform: LetterboxTransform) -> BoxXYXY:
    """The un-padded region of the model-input canvas, xyxy in model-input px.

    Everything outside it is padding and must not be sampled (see `DepthMap.valid`).
    """
    new_w = transform.in_w * transform.scale
    new_h = transform.in_h * transform.scale
    return (
        transform.pad_x,
        transform.pad_y,
        transform.pad_x + new_w,
        transform.pad_y + new_h,
    )
