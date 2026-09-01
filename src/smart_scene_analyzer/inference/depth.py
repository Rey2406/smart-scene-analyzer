"""The reference Depth Anything V2 backend (`DepthBackend`).

**Single responsibility:** be the PyTorch depth model that the exported `.pte` artifact is
proved equivalent to (N20).

**What it returns, stated once and enforced everywhere downstream:** relative inverse
depth. Larger is nearer. There is no unit, there is no scale, and two values are
comparable only when they came from the same image. This model cannot produce a physical
distance and neither can anything built on it — which is why N4 measures pairwise ordering
accuracy rather than a metric error, and why `units_guard.py` blocks the alternative
vocabulary at write time rather than at review time.

**Impure.** It loads weights and imports `transformers`/`timm`. Integration-marked; unit
tests use a double behind `DepthBackend`.

Planned public surface:

    class ReferenceDepthEstimator:
        \"\"\"Implements DepthBackend over a Depth Anything V2 checkpoint.\"\"\"
        def __init__(self, checkpoint: Path, registry_version: str) -> None: ...
        @property
        def registry_version(self) -> str: ...
        def infer(self, model_input: DepthInput) -> DepthMap: ...

The returned `DepthMap` stays in `DEPTH_INPUT` space and is not resampled to `SOURCE`
(architecture R4): fusion maps boxes into this grid instead, because fusion reads a few
hundred samples out of the map and a full-resolution resample would buy nothing for the
cost of roughly a million interpolations on the latency path.
"""
