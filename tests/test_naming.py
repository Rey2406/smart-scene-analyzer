"""The naming contract: no public name anywhere implies a metric unit for depth.

Walks the actual public surface of `src/smart_scene_analyzer` rather than hardcoding
today's module/name list, so a rename that slips past `units_guard.py` (a write-time text
match) still fails here at test time.
"""

from __future__ import annotations

import importlib
import pkgutil

import smart_scene_analyzer

BANNED_SUBSTRINGS = ("meter", "metre", "_mm", "_cm", "distance")


def _iter_project_modules():
    """Every importable module under `smart_scene_analyzer`, recursively."""
    yield smart_scene_analyzer
    for info in pkgutil.walk_packages(
        smart_scene_analyzer.__path__, prefix=f"{smart_scene_analyzer.__name__}."
    ):
        try:
            yield importlib.import_module(info.name)
        except ImportError:
            # A module whose own heavy import (ultralytics, transformers) isn't
            # installed in this environment is not this test's concern — its names are
            # still checked by whichever suite DOES have that extra installed, and this
            # test must not need those extras to run in the default offline suite.
            continue


def test_no_public_name_in_the_project_implies_a_metric_unit():
    offenders: list[str] = []
    for module in _iter_project_modules():
        for name in dir(module):
            if name.startswith("_"):
                continue
            if any(bad in name.lower() for bad in BANNED_SUBSTRINGS):
                offenders.append(f"{module.__name__}.{name}")
    assert offenders == [], f"Metric-unit-implying public names found: {offenders}"


def test_depth_module_docstring_states_the_sign_convention():
    from smart_scene_analyzer import depth

    text = (depth.__doc__ or "") + (depth.estimate_relative_inverse_depth.__doc__ or "")
    lowered = text.lower()
    assert "nearer" in lowered
    assert "not a distance, in any unit" in lowered


def test_fusion_module_docstring_states_units_and_naming():
    from smart_scene_analyzer import fusion

    lowered = (fusion.__doc__ or "").lower()
    assert "relative inverse depth" in lowered
    assert "larger is nearer" in lowered


def test_boxes_are_documented_as_xyxy_in_a_named_space():
    """The contract every `Detection`/`BoxXYXY` consumer relies on: xyxy, absolute pixels,
    and WHICH space — never left implicit."""
    from smart_scene_analyzer import types

    box_doc = (types.BoxXYXY.__doc__ or "") or ""
    # `BoxXYXY` is a type alias; its contract lives in the module docstring and on the
    # dataclasses that carry it (`Detection`, etc.) rather than on the alias itself.
    combined = (types.__doc__ or "") + box_doc + (types.Detection.__doc__ or "")
    lowered = combined.lower()
    assert "xyxy" in lowered
    assert "source" in lowered or "det_input" in lowered or "named" in lowered
