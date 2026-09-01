"""Coordinate spaces and box arithmetic.

**Single responsibility:** own every conversion between the project's coordinate spaces,
so that no conversion happens anywhere else.

This package has no dependency on a model, a file, or a runtime. It is pure arithmetic
over numbers and small arrays, which is what makes it the cheapest thing in the repository
to test — a box, a transform, an expected box — and this is the package where the
project's coordinate bugs would otherwise live.

The four frames on the inference path (`docs/architecture.md` §3):

- `SOURCE` — the decoded working image, display-upright, absolute pixels.
- `DET_INPUT` — the letterboxed detection-model canvas, absolute pixels, own scale/pads.
- `DEPTH_INPUT` — the letterboxed depth-model canvas, absolute pixels, **different**
  scale/pads from `DET_INPUT`.
- `SCREEN` — the image view, density-independent points.

The rule this package exists to enforce: **a conversion between two spaces happens in
exactly one named function whose signature names both spaces.** A box indexed into the
wrong space returns a number — the wrong number, in range — and nothing raises.
"""
