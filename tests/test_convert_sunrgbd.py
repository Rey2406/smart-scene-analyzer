"""Tests for the SUN RGB-D -> YOLO converter.

No GPU, no network, no `.mat` on disk, no images: every fixture here is synthesized.
The taxonomy fixtures are deliberately *not* the real `docs/taxonomy.md`, so that
accepting a new class never breaks these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smart_scene_analyzer.convert_sunrgbd import (
    BoxRejectedError,
    flat_name,
    format_yolo_line,
    parse_taxonomy,
    to_yolo_box,
)

TAXONOMY_OK = """
# Object taxonomy

| ID | Class | Source labels mapped in (exact match) | Objects | Images |
|---|---|---|---|---|
| 0 | `chair` | `chair`, `sofa_chair`, `stool` | 25,032 | ~5,300 |
| 1 | `table` | `table`, `desk` | 9,808 | ~4,400 |
| 2 | `storage` | `cabinet`, `dresser` | 4,325 | ~2,600 |

Some prose that must not be parsed as a row.

| Class | Include | Exclude |
|---|---|---|
| `chair` | Any single-person seat | Multi-person seating |
"""


def _write(tmp_path, text):
    p = tmp_path / "taxonomy.md"
    p.write_text(text)
    return p


# --------------------------------------------------------------- taxonomy parsing


def test_parses_classes_in_id_order(tmp_path):
    tax = parse_taxonomy(_write(tmp_path, TAXONOMY_OK))
    assert tax.classes == ["chair", "table", "storage"]


def test_maps_every_source_label_to_its_class_id(tmp_path):
    tax = parse_taxonomy(_write(tmp_path, TAXONOMY_OK))
    assert tax.id_of("chair") == 0
    assert tax.id_of("sofa_chair") == 0
    assert tax.id_of("stool") == 0
    assert tax.id_of("desk") == 1
    assert tax.id_of("dresser") == 2


def test_matching_is_exact_never_substring(tmp_path):
    """`dresser_mirror` is a mirror and `sofa_chair` is a chair.

    The real dataset contains 169 `dresser_mirror` objects. Substring matching would
    file them under `storage`, and nothing downstream would report it.
    """
    tax = parse_taxonomy(_write(tmp_path, TAXONOMY_OK))
    assert tax.id_of("dresser_mirror") is None
    assert tax.id_of("chair_leg") is None
    assert tax.id_of("sofa_chair") == 0, "must be chair, not sofa"


def test_unmapped_label_returns_none_rather_than_guessing(tmp_path):
    tax = parse_taxonomy(_write(tmp_path, TAXONOMY_OK))
    assert tax.id_of("toilet") is None
    assert tax.id_of("") is None


def test_ignores_the_inclusion_rule_table(tmp_path):
    """The second table has no numeric ID column and must not become a class."""
    tax = parse_taxonomy(_write(tmp_path, TAXONOMY_OK))
    assert len(tax.classes) == 3


def test_rejects_non_contiguous_class_ids(tmp_path):
    bad = TAXONOMY_OK.replace("| 2 | `storage`", "| 5 | `storage`")
    with pytest.raises(ValueError, match="contiguous"):
        parse_taxonomy(_write(tmp_path, bad))


def test_rejects_out_of_order_class_ids(tmp_path):
    """Order is the export contract; a swap must fail loudly, not silently relabel."""
    lines = TAXONOMY_OK.splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("| 0 |"))
    lines[i], lines[i + 1] = lines[i + 1], lines[i]
    with pytest.raises(ValueError, match=r"contiguous|expected"):
        parse_taxonomy(_write(tmp_path, "\n".join(lines)))


def test_rejects_one_source_label_mapped_to_two_classes(tmp_path):
    bad = TAXONOMY_OK.replace("| `cabinet`, `dresser` |", "| `cabinet`, `chair` |")
    with pytest.raises(ValueError, match="mapped to two classes"):
        parse_taxonomy(_write(tmp_path, bad))


def test_rejects_a_file_with_no_class_rows(tmp_path):
    with pytest.raises(ValueError, match="no class rows"):
        parse_taxonomy(_write(tmp_path, "# Object taxonomy\n\nStill an open decision.\n"))


# ------------------------------------------------------------ coordinate conversion


def test_centred_box_converts_to_centre_of_frame():
    cx, cy, nw, nh = to_yolo_box(25, 50, 50, 100, 100, 200)
    assert (cx, cy, nw, nh) == (0.5, 0.5, 0.5, 0.5)


def test_top_left_origin_is_top_left_not_bottom_left():
    """A y-flip is the error this project is most likely to make and least likely to
    notice: it raises nothing and trains a worse model."""
    _cx, cy, _nw, nh = to_yolo_box(0, 0, 10, 20, 100, 100)
    assert cy == pytest.approx(0.1), "a box at y=0 must land near the TOP of the frame"
    assert nh == pytest.approx(0.2)


def test_xywh_is_not_read_as_xyxy():
    """`w`/`h` are extents, not far corners. Read as xyxy this box is degenerate."""
    cx, _cy, nw, _nh = to_yolo_box(80, 10, 40, 10, 200, 200)
    assert nw == pytest.approx(0.2), "width 40 of 200"
    assert cx == pytest.approx(0.5), "centre at x=100 of 200"


def test_box_overhanging_the_frame_is_clipped_not_rejected():
    cx, _cy, nw, _nh = to_yolo_box(-10, 0, 60, 10, 100, 100)
    assert nw == pytest.approx(0.5), "clipped to 0..50"
    assert cx == pytest.approx(0.25)


def test_full_frame_box_is_allowed():
    cx, cy, nw, nh = to_yolo_box(0, 0, 100, 100, 100, 100)
    assert (cx, cy, nw, nh) == (0.5, 0.5, 1.0, 1.0)


@pytest.mark.parametrize(
    ("box", "reason"),
    [
        ((10, 10, 0, 10), "non-positive width/height"),
        ((10, 10, 10, -5), "non-positive width/height"),
        ((200, 10, 10, 10), "box entirely outside image"),
        ((-50, 10, 10, 10), "box entirely outside image"),
    ],
)
def test_bad_boxes_are_rejected_with_an_itemized_reason(box, reason):
    with pytest.raises(BoxRejectedError) as exc:
        to_yolo_box(*box, 100, 100)
    assert exc.value.reason == reason


def test_rejects_degenerate_image_dimensions():
    with pytest.raises(BoxRejectedError, match="non-positive dimensions"):
        to_yolo_box(1, 1, 1, 1, 0, 100)


def test_every_output_coordinate_is_normalized():
    for args in [(0, 0, 1, 1, 640, 480), (639, 479, 1, 1, 640, 480), (10, 10, 600, 400, 640, 480)]:
        for v in to_yolo_box(*args):
            assert 0.0 <= v <= 1.0


# ------------------------------------------------------------------- line formatting


def _edges(line: str) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) implied by a rendered label line."""
    _cid, cx, cy, nw, nh = line.split()
    cx, cy, nw, nh = float(cx), float(cy), float(nw), float(nh)
    return cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2


def test_line_has_class_id_and_four_coordinates():
    assert format_yolo_line(3, 0.5, 0.5, 0.25, 0.25) == "3 0.500000 0.500000 0.250000 0.250000"


@pytest.mark.parametrize(
    "box",
    [
        (0.5, 0.5, 1.0, 1.0),  # full frame
        (0.0000005, 0.5, 0.000001, 0.5),  # hugging the left edge
        (0.9999995, 0.5, 0.000001, 0.5),  # hugging the right edge
        (0.5, 0.9999995, 0.5, 0.000001),  # hugging the bottom edge
        (1 / 3, 1 / 7, 2 / 3, 2 / 7),  # non-terminating decimals
    ],
)
def test_rounded_box_never_escapes_the_frame(box):
    """Roboflow answers an out-of-frame annotation with `annotation:trimmed`.

    Rounding a box that sits exactly on an edge is what causes it — 3,402 of 51,019
    boxes on this dataset before this was floored rather than rounded.
    """
    left, top, right, bottom = _edges(format_yolo_line(0, *box))
    assert left >= 0.0, f"left edge {left} < 0"
    assert top >= 0.0, f"top edge {top} < 0"
    assert right <= 1.0, f"right edge {right} > 1"
    assert bottom <= 1.0, f"bottom edge {bottom} > 1"


def test_extent_is_never_rounded_away_to_zero():
    for cx, cy, nw, nh in [(0.5, 0.5, 1e-12, 1e-12), (0.0000001, 0.0000001, 1e-9, 1e-9)]:
        _cid, _x, _y, w, h = format_yolo_line(0, cx, cy, nw, nh).split()
        assert float(w) > 0.0
        assert float(h) > 0.0


def test_shrink_is_at_most_one_unit_in_the_last_place():
    """The fix must not visibly move boxes — it may only shave a rounding artifact."""
    _cid, _x, _y, w, h = format_yolo_line(0, 0.5, 0.5, 0.123456789, 0.987654321).split()
    assert abs(float(w) - 0.123456789) <= 1e-6
    assert abs(float(h) - 0.987654321) <= 1e-6


# --------------------------------------------------- the real taxonomy's ID contract


def test_real_taxonomy_is_numbered_alphabetically():
    """Roboflow sorts class names alphabetically on export, and it is not configurable.

    If `docs/taxonomy.md` is numbered any other way, the document says `chair = 0` while
    every label file in the export says `bed = 0`. Nothing raises: the app detects
    correctly, draws boxes correctly, and names every one of them wrong. See ADR 0003.

    This test exists because the failure is invisible without it — it was introduced
    once already and caught only by reading the lesson's warning.
    """
    tax = parse_taxonomy(Path(__file__).resolve().parents[1] / "docs" / "taxonomy.md")
    assert tax.classes == sorted(tax.classes), (
        f"docs/taxonomy.md must be numbered alphabetically to match the export. "
        f"Got {tax.classes}, expected {sorted(tax.classes)}"
    )


def test_real_taxonomy_has_no_duplicate_class_names():
    tax = parse_taxonomy(Path(__file__).resolve().parents[1] / "docs" / "taxonomy.md")
    assert len(set(tax.classes)) == len(tax.classes)


# ----------------------------------------------------------------- upload filenames


def test_flat_name_flattens_the_session_path():
    assert (
        flat_name("SUNRGBD/kv2/align_kv2/2014-12-18_11-31-54_260595134347", "0000144.jpg")
        == "kv2_align_kv2_2014-12-18_11-31-54_260595134347_0000144.jpg"
    )


def test_flat_name_disambiguates_reused_frame_filenames():
    """SUN RGB-D's 8,332 RGB files share only 3,417 distinct basenames.

    Keying on the basename alone would collapse 4,915 images onto each other, and the
    upload would succeed while carrying 41% of the dataset.
    """
    a = flat_name("SUNRGBD/kv1/NYUdata/NYU0428", "0000103.jpg")
    b = flat_name("SUNRGBD/kv2/kinect2data/000002-resize", "0000103.jpg")
    assert a != b


def test_flat_name_drops_only_the_leading_dataset_component():
    name = flat_name("SUNRGBD/xtion/sun3ddata/mit_32_bar", "0011913.jpg")
    assert not name.startswith("SUNRGBD")
    assert name == "xtion_sun3ddata_mit_32_bar_0011913.jpg"
