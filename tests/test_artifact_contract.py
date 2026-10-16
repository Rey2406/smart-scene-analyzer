"""The artifact contract (C11): label order traced to its one source of truth, and the
zero/degenerate-input boundary cases for the real, non-stub entry points that exist today.

`docs/taxonomy.md` is the single source of truth for label order (ADR 0003). `data/v1/data.yaml`
and `app/assets/models/model_contract.json` are both *derived* from it and must never be
compared against each other directly — only against the taxonomy, so a drift between the
two derived copies still gets caught even if nobody happened to diff them against each
other.

Detection-specific boundary cases from this suite's own brief (zero detections from a raw
model output, grayscale/1x1 input to the detector) depend on `inference.decode` and
`inference.detector`, which are still "planned public surface" stubs as of this suite —
not fabricated here. The equivalent cases that already have a real implementation
(`fusion.fuse([])`, `depth.estimate_relative_inverse_depth` on unusual inputs) are covered
in `test_fusion.py` and below, respectively.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = PROJECT_ROOT / "docs" / "taxonomy.md"
DATA_YAML_PATH = PROJECT_ROOT / "data" / "v1" / "data.yaml"
MANIFEST_PATH = PROJECT_ROOT / "app" / "assets" / "models" / "model_contract.json"


def _taxonomy_labels() -> list[str]:
    """Parse the `| ID | Class | ... |` table under taxonomy.md's `## Class list` heading.

    Anchored to that heading by name, same as `scripts/verify_export.py` — without it, any
    other `| <int> | <name> |`-shaped table in the file could be matched instead.
    """
    text = TAXONOMY_PATH.read_text()
    after_heading = text.split("## Class list", 1)[1]
    rows = re.findall(r"^\|\s*(\d+)\s*\|\s*`(\w+)`\s*\|", after_heading, re.MULTILINE)
    by_id = {int(i): name for i, name in rows}
    assert by_id, "No class rows parsed out of docs/taxonomy.md's Class list table."
    return [by_id[i] for i in sorted(by_id)]


def test_taxonomy_is_internally_consistent():
    """Always runnable: no generated or gitignored artifact required."""
    labels = _taxonomy_labels()
    assert labels == sorted(labels), "docs/taxonomy.md's own ordering rule is alphabetical."
    assert len(labels) == len(set(labels)), "Duplicate class name in docs/taxonomy.md."


def test_data_yaml_label_order_matches_taxonomy():
    if not DATA_YAML_PATH.is_file():
        pytest.skip(f"{DATA_YAML_PATH} is gitignored and not present in this checkout.")
    with DATA_YAML_PATH.open() as f:
        data_cfg = yaml.safe_load(f)
    data_labels = [data_cfg["names"][i] for i in sorted(data_cfg["names"])]
    assert data_labels == _taxonomy_labels(), (
        "data/v1/data.yaml's label order has drifted from docs/taxonomy.md, its single "
        "source of truth (ADR 0003)."
    )


def test_exported_manifest_label_order_matches_taxonomy():
    if not MANIFEST_PATH.is_file():
        pytest.skip(f"{MANIFEST_PATH} not present — detection has not been exported yet.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    detector = manifest.get("detector")
    if detector is None:
        pytest.skip("model_contract.json has no detector section yet.")
    assert detector["labels"] == _taxonomy_labels(), (
        "The exported artifact's label order has drifted from docs/taxonomy.md — this is "
        "the highest-risk field in the project (C11): an app built against it draws "
        "correct boxes with the wrong names, and nothing at runtime catches it."
    )


# --------------------------------------------------------- real-entry-point boundary cases


def test_depth_handles_grayscale_input_not_a_crash():
    from smart_scene_analyzer import depth

    gray = np.zeros((16, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxWx3"):
        depth.estimate_relative_inverse_depth(
            gray,
            loader=lambda _device: lambda _img: np.zeros((14, 14), dtype=np.float32),
        )


def test_depth_handles_1x1_input_not_a_crash():
    from smart_scene_analyzer import depth

    tiny = np.zeros((1, 1, 3), dtype=np.uint8)
    out = depth.estimate_relative_inverse_depth(
        tiny,
        loader=lambda _device: lambda _img: np.zeros((14, 14), dtype=np.float32),
    )
    assert out.shape == (1, 1)


def test_fuse_with_zero_detections_is_an_empty_list_not_none_or_an_exception(fake_depth_map):
    from smart_scene_analyzer.fusion import fuse

    result = fuse([], fake_depth_map)
    assert result == []
    assert result is not None
