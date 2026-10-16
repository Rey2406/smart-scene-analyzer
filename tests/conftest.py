"""Shared fixtures for the offline suite (Lesson 04, step 5; `.claude/skills/offline-suite`).

No GPU, no network, no weights on disk, in seconds. Every fixture here is generated in
code at test time — **nothing binary is committed**.

Mock at the model boundary, not at the fusion boundary: `mock_models` fakes the expensive
dependency (a loaded model), never the pure logic (`fusion.fuse`) that is this suite's
actual subject.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from smart_scene_analyzer.types import DepthMap, Detection, LetterboxTransform


@pytest.fixture
def sample_image() -> Image.Image:
    """An RGB PIL image, generated at test time — never a committed binary."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), mode="RGB")


@pytest.fixture
def sample_image_nonsquare() -> Image.Image:
    """Same, at a different aspect ratio — catches code that silently assumes square."""
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (13, 41, 3), dtype=np.uint8), mode="RGB")


@pytest.fixture
def fake_depth_map() -> DepthMap:
    """A `DepthMap` over a 20x10 **known gradient**: value == column index.

    Column index is the value, so the median of any box is the median of the column
    indices it spans — computable on paper. The transform is identity (`SOURCE` ==
    `DEPTH_INPUT`, no padding) so fusion tests exercise fusion's own logic without also
    exercising `geometry.letterbox`'s arithmetic, which has its own coverage.
    """
    h, w = 20, 10
    values = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    transform = LetterboxTransform(
        scale=1.0, pad_x=0.0, pad_y=0.0, in_w=w, in_h=h, out_w=w, out_h=h
    )
    return DepthMap(values=values, transform=transform, valid=(0.0, 0.0, float(w), float(h)))


@pytest.fixture
def fake_detections() -> list[Detection]:
    """Canned detections with known boxes, over `fake_depth_map`'s gradient.

    `near` spans columns 6-9 (high value, nearer); `far` spans columns 0-3 (low value,
    farther) — chosen so the expected near/far order is obvious without running anything.
    """
    return [
        Detection(bbox=(6.0, 0.0, 10.0, 10.0), class_id=0, label="near", confidence=0.9),
        Detection(bbox=(0.0, 0.0, 4.0, 10.0), class_id=1, label="far", confidence=0.8),
    ]


class _FakeDetectorBackend:
    """Satisfies `inference.protocols.DetectorBackend` structurally, with a canned
    response. No weights load, no GPU touched."""

    def __init__(self, raw_output: object, registry_version: str = "fake-v0") -> None:
        self._raw_output = raw_output
        self._registry_version = registry_version

    @property
    def registry_version(self) -> str:
        return self._registry_version

    def infer(self, model_input: object) -> object:
        return self._raw_output


@pytest.fixture
def mock_models():
    """Patches the expensive model boundary so no weights load and no GPU is touched.

    `depth.py`'s loader seam is real (Lesson 03) and mockable today; a `DetectorBackend`
    double is provided too, satisfying the structural protocol in `inference.protocols`,
    ready for when `inference.detector`'s reference implementation (currently a stub)
    exists to be mocked in place of.
    """

    def fake_depth_loader(raster: np.ndarray):
        def build(_device: str):
            def run(_image):
                return raster

            return run

        return build

    def fake_detector(raw_output: object, registry_version: str = "fake-v0"):
        return _FakeDetectorBackend(raw_output, registry_version)

    return {"depth_loader": fake_depth_loader, "detector": fake_detector}
