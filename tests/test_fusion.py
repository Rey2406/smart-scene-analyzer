"""Offline tests for the Scene Understanding Layer (`fusion.py`, stage S6).

No GPU, no network, no weights on disk — `fuse` is pure by construction, so there is no
excuse not to test it exhaustively. The known-gradient fixture (`fake_depth_map`) makes
every assertion exact rather than "seems plausible": the expected median for a given box
is the median of the column indices it spans, computable by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from smart_scene_analyzer.fusion import (
    MIXED_REGION_SPREAD_THRESHOLD,
    fuse,
    fuse_detections_with_depth,
    reduce_box,
    robust_range,
    sort_near_to_far,
)
from smart_scene_analyzer.types import DepthMap, Detection, LetterboxTransform

FUSION_CASES_PATH = Path(__file__).parent / "fixtures" / "fusion_cases.json"


# ------------------------------------------------- the test worth arguing about


def test_fusion_actually_uses_the_depth_map(fake_detections, fake_depth_map):
    """A fusion function that ignores its depth argument passes everything else.

    Every other test in this file would pass on a `fuse()` that returns a constant: the
    shapes are right, the units are right, and a single-fixture ordering assertion can be
    satisfied by accident. This one cannot — it runs the SAME detections over two maps
    that disagree almost everywhere.
    """
    first = fuse(fake_detections, fake_depth_map)
    inverted_values = fake_depth_map.values.max() - fake_depth_map.values
    inverted_map = DepthMap(
        values=inverted_values, transform=fake_depth_map.transform, valid=fake_depth_map.valid
    )
    second = fuse(fake_detections, inverted_map)
    assert [d.relative_depth for d in first] != [d.relative_depth for d in second]


# ------------------------------------------------------------- exact values on a gradient


def test_correct_depth_value_for_a_known_region(fake_depth_map):
    # Columns 2,3,4,5 of a column==value gradient -> median 3.5, same as depth.py's own
    # known-gradient convention.
    box = (2.0, 0.0, 6.0, 5.0)
    stat = reduce_box(fake_depth_map, box, inset_fraction=0.0, max_samples=1024)
    assert stat.median == pytest.approx(3.5)


def test_near_to_far_ordering_matches_the_gradient(fake_detections, fake_depth_map):
    fused = fuse(fake_detections, fake_depth_map)
    near, far = fused  # fake_detections is [near (cols 6-9), far (cols 0-3)]
    assert near.relative_depth > far.relative_depth
    assert near.depth_rank == 0
    assert far.depth_rank == 1


def test_robust_range_is_the_2nd_98th_percentile(fake_depth_map):
    lo, hi = robust_range(fake_depth_map)
    # 20x10 gradient, columns 0..9 each repeated 20 times: 2nd/98th percentile of a
    # uniform discrete distribution over [0..9] lands close to the extremes but not
    # exactly at them.
    assert 0.0 <= lo < 1.0
    assert 8.0 < hi <= 9.0


# ------------------------------------------------------------------ degenerate box cases


def test_zero_area_box_is_degenerate(fake_depth_map):
    stat = reduce_box(fake_depth_map, (5.0, 5.0, 5.0, 8.0), inset_fraction=0.125, max_samples=1024)
    assert stat.median is None
    assert stat.spread is None
    assert stat.sample_count == 0


def test_box_partly_outside_is_clipped_not_an_error(fake_depth_map):
    # Columns -50..4, clipped to the valid 0..10 content rect -> columns 0..3.
    stat = reduce_box(
        fake_depth_map, (-50.0, -50.0, 4.0, 5.0), inset_fraction=0.0, max_samples=1024
    )
    assert stat.median == pytest.approx(1.5)  # median of [0,1,2,3]


def test_box_entirely_outside_is_degenerate(fake_depth_map):
    stat = reduce_box(
        fake_depth_map, (100.0, 0.0, 150.0, 5.0), inset_fraction=0.125, max_samples=1024
    )
    assert stat.median is None
    assert stat.sample_count == 0


def test_empty_after_clipping_is_degenerate(fake_depth_map):
    # A box that touches the valid rect at a single edge clips to zero width.
    stat = reduce_box(
        fake_depth_map, (10.0, 0.0, 15.0, 5.0), inset_fraction=0.125, max_samples=1024
    )
    assert stat.median is None
    assert stat.sample_count == 0


def test_fuse_with_degenerate_box_reports_none_not_a_crash(fake_depth_map):
    degenerate = Detection(bbox=(5.0, 5.0, 5.0, 8.0), class_id=0, label="zero", confidence=0.5)
    fused = fuse([degenerate], fake_depth_map)
    assert fused[0].relative_depth is None
    assert fused[0].relative_depth_normalized is None
    assert fused[0].depth_rank is None
    assert fused[0].depth_spread is None
    assert fused[0].depth_sample_count == 0


# ----------------------------------------------------------------------------- boundaries


def test_no_detections_is_an_empty_list_not_an_error(fake_depth_map):
    assert fuse([], fake_depth_map) == []


def test_no_depth_map_degrades_every_detection_the_same_way(fake_detections):
    fused = fuse(fake_detections, None)
    assert len(fused) == len(fake_detections)
    for d in fused:
        assert d.relative_depth is None
        assert d.relative_depth_normalized is None
        assert d.depth_rank is None
        assert d.depth_sample_count == 0


def test_a_valid_rect_that_exceeds_the_array_is_clipped_not_crashed():
    """`depth_map.valid` is normally inside the array's bounds, but a rounding-induced
    1px overshoot at an edge must not crash or silently resize the array — it clips."""
    values = np.tile(np.arange(10, dtype=np.float32), (20, 1))
    transform = LetterboxTransform(
        scale=1.0, pad_x=0.0, pad_y=0.0, in_w=10, in_h=20, out_w=10, out_h=20
    )
    overshooting = DepthMap(values=values, transform=transform, valid=(0.0, 0.0, 11.0, 21.0))
    stat = reduce_box(overshooting, (2.0, 0.0, 6.0, 5.0), inset_fraction=0.0, max_samples=1024)
    assert stat.median == pytest.approx(3.5)  # unaffected — the box itself is in-bounds
    assert values.shape == (20, 10)  # the array was never touched/resized


def test_ordering_is_stable_across_different_depth_map_resolutions():
    """The 'same scene', letterboxed to two different model-input canvases, must agree
    on which detection is nearer — `fuse` must not be sensitive to `DEPTH_SIZE` itself."""

    def build(out_w: int, out_h: int) -> DepthMap:
        values = np.tile(np.arange(out_w, dtype=np.float32), (out_h, 1))
        transform = LetterboxTransform(
            scale=1.0, pad_x=0.0, pad_y=0.0, in_w=out_w, in_h=out_h, out_w=out_w, out_h=out_h
        )
        return DepthMap(
            values=values, transform=transform, valid=(0.0, 0.0, float(out_w), float(out_h))
        )

    small = build(10, 10)
    large = build(40, 40)

    # Position "near" at the high-value end and "far" at the low-value end in both canvases.
    near_small = Detection(bbox=(8.0, 0.0, 10.0, 10.0), class_id=0, label="near", confidence=0.9)
    far_small = Detection(bbox=(0.0, 0.0, 2.0, 10.0), class_id=1, label="far", confidence=0.8)
    near_large = Detection(bbox=(32.0, 0.0, 40.0, 40.0), class_id=0, label="near", confidence=0.9)
    far_large = Detection(bbox=(0.0, 0.0, 8.0, 40.0), class_id=1, label="far", confidence=0.8)

    fused_small = fuse([near_small, far_small], small)
    fused_large = fuse([near_large, far_large], large)

    assert fused_small[0].depth_rank == 0 and fused_small[1].depth_rank == 1
    assert fused_large[0].depth_rank == 0 and fused_large[1].depth_rank == 1


# -------------------------------------------------------------------- depth_quality flags


def test_quality_is_ok_for_a_clean_moderate_box(fake_depth_map):
    det = Detection(bbox=(2.0, 0.0, 6.0, 5.0), class_id=0, label="ok", confidence=0.9)
    fused = fuse([det], fake_depth_map)
    assert fused[0].depth_quality == "ok"


def test_quality_is_unavailable_when_depth_map_is_none(fake_detections):
    fused = fuse(fake_detections, None)
    assert all(d.depth_quality == "unavailable" for d in fused)


def test_quality_is_unavailable_for_a_degenerate_box(fake_depth_map):
    zero = Detection(bbox=(5.0, 5.0, 5.0, 8.0), class_id=0, label="zero", confidence=0.5)
    fused = fuse([zero], fake_depth_map)
    assert fused[0].depth_quality == "unavailable"


def test_quality_is_fallback_full_box_when_inset_would_collapse(fake_depth_map):
    """A 1x1 box: insetting by the default 0.125 fraction would shrink it below 1px on
    each side, so `reduce_box` falls back to the un-inset box — and `fuse` must report
    that it did, not silently report `ok`."""
    tiny = Detection(bbox=(3.0, 3.0, 4.0, 4.0), class_id=0, label="tiny", confidence=0.9)
    fused = fuse([tiny], fake_depth_map)
    assert fused[0].depth_quality == "fallback_full_box"
    assert fused[0].relative_depth == pytest.approx(3.0)  # the single pixel column 3


def test_quality_is_mixed_region_above_the_spread_threshold(fake_depth_map):
    """A box spanning the full gradient straddles every depth value in the image —
    exactly the discontinuity-straddling case `mixed_region` exists to disclose."""
    wide = Detection(bbox=(0.0, 0.0, 10.0, 20.0), class_id=0, label="wide", confidence=0.9)
    fused = fuse([wide], fake_depth_map)
    assert fused[0].depth_spread >= MIXED_REGION_SPREAD_THRESHOLD
    assert fused[0].depth_quality == "mixed_region"


# ------------------------------------------- fuse_detections_with_depth / sort_near_to_far


def test_fuse_detections_with_depth_matches_fuse_on_the_wrapped_map(fake_depth_map):
    """The raw-array convenience wrapper must agree exactly with calling `fuse` on the
    equivalent `DepthMap` — it is a bridge, not a second implementation."""
    det = Detection(bbox=(2.0, 0.0, 6.0, 5.0), class_id=0, label="ok", confidence=0.9)
    h, w = fake_depth_map.values.shape

    direct = fuse([det], fake_depth_map)
    wrapped = fuse_detections_with_depth(
        [det], fake_depth_map.values, image_height=h, image_width=w
    )
    assert wrapped[0].relative_depth == direct[0].relative_depth
    assert wrapped[0].depth_quality == direct[0].depth_quality


def test_fuse_detections_with_depth_rejects_a_mismatched_shape(fake_depth_map):
    det = Detection(bbox=(2.0, 0.0, 6.0, 5.0), class_id=0, label="ok", confidence=0.9)
    with pytest.raises(ValueError, match="does not match"):
        fuse_detections_with_depth([det], fake_depth_map.values, image_height=999, image_width=999)


def test_fuse_detections_with_depth_passes_through_no_depth_map():
    det = Detection(bbox=(2.0, 0.0, 6.0, 5.0), class_id=0, label="ok", confidence=0.9)
    fused = fuse_detections_with_depth([det], None, image_height=20, image_width=10)
    assert fused[0].depth_quality == "unavailable"


def test_sort_near_to_far_puts_unranked_detections_last(fake_depth_map):
    zero = Detection(bbox=(5.0, 5.0, 5.0, 8.0), class_id=0, label="zero", confidence=0.5)
    ok_det = Detection(bbox=(2.0, 0.0, 6.0, 5.0), class_id=1, label="ok", confidence=0.9)
    fused = fuse([zero, ok_det], fake_depth_map)  # zero first, deliberately out of order

    ranked = sort_near_to_far(fused)
    assert [d.detection.label for d in ranked] == ["ok", "zero"]
    assert ranked[0].depth_rank == 0
    assert ranked[1].depth_rank is None


# ------------------------------------------------- the shared fixture (contract C12, N21)


def _load_fusion_cases() -> list[dict]:
    return json.loads(FUSION_CASES_PATH.read_text())


def _parse_depth_values(raw: list[list[float | str]]) -> np.ndarray:
    """Plain JSON has no literal NaN/Infinity (Python's `json` module accepts the bare
    tokens as a non-standard extension, but JS `JSON.parse` rejects them outright, and
    this fixture file must load in both languages). Non-finite depth samples are
    therefore encoded as the strings "NaN" / "Infinity" / "-Infinity" instead of bare
    numbers; this converts them back before building the array. The TypeScript loader
    applies the same convention to the same file.
    """
    sentinels = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}
    return np.array(
        [[sentinels.get(v, v) if isinstance(v, str) else v for v in row] for row in raw],
        dtype=np.float32,
    )


@pytest.mark.parametrize("case", _load_fusion_cases(), ids=lambda c: c["name"])
def test_shared_fusion_fixture_matches_python(case):
    """Loads the SAME plain-JSON file Lesson 05's TypeScript suite will load, so the two
    implementations can never silently diverge — a disagreement here is the finding, never
    a reason to adjust the fixture (`docs/architecture.md` C12)."""
    values = _parse_depth_values(case["depth_map"]["values"])
    t = case["transform"]
    transform = LetterboxTransform(
        scale=t["scale"],
        pad_x=t["pad_x"],
        pad_y=t["pad_y"],
        in_w=t["in_w"],
        in_h=t["in_h"],
        out_w=t["out_w"],
        out_h=t["out_h"],
    )
    # `valid` is deliberately not stored in the fixture (matches contract C12): it is
    # fully derivable from `transform` alone, and storing it separately is exactly how
    # the earlier draft of this fixture went quietly self-inconsistent.
    from smart_scene_analyzer.geometry.letterbox import content_rect

    depth_map = DepthMap(values=values, transform=transform, valid=content_rect(transform))
    detections = [
        Detection(
            bbox=tuple(d["bbox"]), class_id=d["class_id"], label="", confidence=d["confidence"]
        )
        for d in case["detections"]
    ]

    fused = fuse(detections, depth_map)
    tol = case["tolerance"]["relative_depth"]
    norm_tol = case["tolerance"]["normalized"]

    for actual, expected in zip(fused, case["expected"]["detections"], strict=True):
        if expected["relative_depth"] is None:
            assert actual.relative_depth is None
        else:
            assert actual.relative_depth == pytest.approx(expected["relative_depth"], abs=tol)
        if expected["relative_depth_normalized"] is None:
            assert actual.relative_depth_normalized is None
        else:
            assert actual.relative_depth_normalized == pytest.approx(
                expected["relative_depth_normalized"], abs=norm_tol
            )
        assert actual.depth_rank == expected["depth_rank"]
        assert actual.depth_sample_count == expected["depth_sample_count"]
        if "depth_quality" in expected:
            assert actual.depth_quality == expected["depth_quality"]
