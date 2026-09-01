"""Smart Scene Analyzer — the Python reference implementation, trainer, and exporter.

**Single responsibility of this package:** be the thing the handset is checked against.

Nothing here serves a request. Per ADR 0001 the product runs entirely on the device
(`app/`); this package trains the two models, exports the artifacts the app bundles, and
remains the reference implementation that the exported artifacts (N20) and the TypeScript
fusion port (N21) must agree with on shared fixtures.

Design of record: `docs/architecture.md`. Contracts: `smart_scene_analyzer.types`.

Two conventions hold everywhere below this package, and violating either produces a
plausible wrong number rather than an error:

- **Bounding boxes are `xyxy` in absolute pixels of a NAMED coordinate space.** The spaces
  are `SOURCE`, `DET_INPUT`, `DEPTH_INPUT`, and `SCREEN` (`geometry.spaces`). A conversion
  between two of them happens in exactly one named function that names both.
- **Depth is relative inverse depth.** Larger is nearer. There is no unit, there is no
  scale, and values are comparable only within one image. The system cannot state a
  physical distance and no identifier, docstring, comment, or UI string may imply it can.
"""

__version__ = "0.1.0"
