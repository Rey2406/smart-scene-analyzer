"""`model_contract.json` — write it, read it, validate it (contract C11).

**Single responsibility:** be the single written statement of everything the app assumes
about the artifacts, so that those assumptions are reviewable instead of implicit.

The export **defines** this; the app **consumes** it. Every field below is fixed on this
side of the boundary and taken on faith on the other:

- input tensor shape, dtype, and layout, per model — `[1, 640, 640, 3]` NHWC is not
  `[1, 3, 640, 640]` NCHW, and a swap produces plausible garbage rather than an error;
- the quantization scale and zero point for the int8 detection input;
- channel normalization constants;
- the letterbox pad value;
- output tensor count, order, shape, dtype, box encoding (`cxcywh` or `xyxy`), and whether
  coordinates are normalized;
- **the ordered label list.** This is the highest-risk field in the project. An app built
  against a stale order runs, draws correct boxes, and names every one of them wrong, and
  no test on either side catches it. `sha256` plus `artifact_drift.py` is the defence;
- the default confidence and IoU thresholds and the per-image detection cap, so that the
  app and the reference use one set of numbers rather than two constants in two languages
  (assumptions A7, A8);
- the model registry version for each model, which is the only on-device source for F7 —
  there is no service to ask;
- `depth.output.semantics`, a literal string asserting that the depth output is relative
  inverse depth with larger meaning nearer. The app checks it at load, so that an export
  which ever flips polarity fails loudly instead of rendering every scene inside out.

The label list's contents are an **open decision** (the taxonomy, Lesson 02). This module
records whatever order it is given and computes nothing from its length.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class QuantParams(BaseModel):
    """int8 affine quantization: `real_value = scale * (quantized_value - zero_point)`."""

    scale: float
    zero_point: int


class DetectorInputSpec(BaseModel):
    shape: list[int]
    dtype: str
    layout: Literal["NHWC", "NCHW"]
    quant: QuantParams | None = None
    normalization: str
    letterbox_pad_value: list[int]


class DetectorOutputSpec(BaseModel):
    index: int
    name: str
    shape: list[int]
    dtype: str
    encoding: Literal["cxcywh", "xyxy"]
    normalized: bool


class DetectorManifest(BaseModel):
    file: str
    sha256: str
    registry_version: str
    input: DetectorInputSpec
    outputs: list[DetectorOutputSpec]
    labels: list[str]
    default_conf_threshold: float
    default_iou_threshold: float
    max_detections: int


class DepthInputSpec(BaseModel):
    shape: list[int]
    dtype: str
    layout: Literal["NHWC", "NCHW"]
    normalization: str


class DepthOutputSpec(BaseModel):
    shape: list[int]
    dtype: str
    semantics: Literal["relative-inverse-depth-larger-is-nearer"]


class DepthManifest(BaseModel):
    file: str
    sha256: str
    registry_version: str
    input: DepthInputSpec
    output: DepthOutputSpec


class ArtifactManifest(BaseModel):
    """The schema of C11. `detector` and/or `depth` may be absent on a partial export —
    `scripts/export.py --detection` writes only the detector half, merging with whatever
    depth half an earlier run already wrote."""

    detector: DetectorManifest | None = None
    depth: DepthManifest | None = None
    exported_at: str
    exporter_version: str
    dataset_version: str | None = None


def sha256_of(path: Path) -> str:
    """The file's sha256, read in chunks rather than loaded whole (artifacts are tens of MB)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(manifest: ArtifactManifest, path: Path) -> None:
    path.write_text(manifest.model_dump_json(indent=2, exclude_none=True) + "\n")


def read(path: Path) -> ArtifactManifest:
    return ArtifactManifest.model_validate_json(path.read_text())


def verify_files(manifest: ArtifactManifest, models_dir: Path) -> list[str]:
    """Human-readable mismatches: a missing file, or a sha256 that does not match the
    manifest. Empty list means the bundled artifacts are the ones described."""
    problems: list[str] = []
    for name, sub in (("detector", manifest.detector), ("depth", manifest.depth)):
        if sub is None:
            continue
        p = models_dir / sub.file
        if not p.is_file():
            problems.append(f"{name}: missing file {p}")
            continue
        actual = sha256_of(p)
        if actual != sub.sha256:
            problems.append(
                f"{name}: sha256 mismatch for {p} "
                f"(manifest {sub.sha256[:12]}… != actual {actual[:12]}…)"
            )
    return problems
