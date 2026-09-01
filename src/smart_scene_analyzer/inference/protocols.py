"""The model seam: what a backend is, expressed as tensors in and tensors out.

**Single responsibility:** describe a model backend narrowly enough that a test double is
a five-line class, and broadly enough that the real thing fits.

This is the module that makes `uv run pytest -m "not integration"` possible with no GPU,
no network, and no weights on disk. Two rules keep it that way:

- **No file paths, no sessions, no device handles in a signature.** A backend is
  constructed however its implementation likes; the protocol only says what it does.
- **Arrays in, arrays out.** A double returns a canned array loaded from the shared JSON
  fixtures (`smart_scene_analyzer.fixtures`), and because these are structural protocols
  the double imports nothing from this package.

These are also the interfaces the open "one runtime or two" decision moves behind. Either
consolidation branch — detection to ExecuTorch, or depth to TFLite — replaces an
implementation here and leaves every data contract intact.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from smart_scene_analyzer.types import DepthInput, DepthMap, DetectionInput, RawDetectorOutput


@runtime_checkable
class DetectorBackend(Protocol):
    """Runs the detection model. Stage S3.

    Implementations: the reference YOLO11 in `inference.detector` (integration-marked),
    the exported int8 artifact in `export.parity`, and a fixture-backed double in tests.
    """

    @property
    def registry_version(self) -> str:
        """The model registry version this backend is serving. Reported by F7."""
        ...

    def infer(self, model_input: DetectionInput) -> RawDetectorOutput:
        """Run detection on one image.

        Args:
            model_input: `[1, DET_SIZE, DET_SIZE, 3]` NHWC in `DET_INPUT` space, dtype and
                quantization fixed by the export (contract C3).

        Returns:
            The model's raw output tensors, undecoded (contract C5). Tensor count, order,
            shape, box encoding, and whether coordinates are normalized are stated in the
            artifact manifest, not assumed here.
        """
        ...


@runtime_checkable
class DepthBackend(Protocol):
    """Runs the monocular depth model. Stage S4.

    The output is **relative inverse depth**: larger is nearer, the scale is arbitrary,
    and values are comparable only within one image. There is no unit and no conversion to
    a physical distance.
    """

    @property
    def registry_version(self) -> str:
        """The model registry version this backend is serving. Reported by F7."""
        ...

    def infer(self, model_input: DepthInput) -> DepthMap:
        """Run monocular depth estimation on one image.

        Args:
            model_input: `[1, 3, DEPTH_SIZE, DEPTH_SIZE]` float32 NCHW in `DEPTH_INPUT`
                space (contract C4).

        Returns:
            A `DepthMap`: float32 `[DEPTH_SIZE, DEPTH_SIZE]` of relative inverse depth in
            `DEPTH_INPUT` space, larger is nearer, no unit and no scale — carrying the
            `SOURCE` -> `DEPTH_INPUT` transform and the un-padded content rectangle, so
            that the fusion layer never samples a padding pixel (contract C8).
        """
        ...
