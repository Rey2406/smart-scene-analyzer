"""Tests for the annotation uploader's pure parts.

No network and no credentials: `roboflow` is imported inside `upload()`, so nothing here
touches the platform. The two functions under test are the ones that would silently put
annotations on the wrong image if they were wrong.
"""

from __future__ import annotations

import pytest

from smart_scene_analyzer.upload_annotations import build_labelmap, parse_upload_log

LOG = """\
loading Roboflow project...
Uploading to existing project ws/smart-scene-analyzer-2026-09-08
  0%|          | 0/8332 [00:00<?, ?it/s]100%|##########| 8332/8332 [00:00<00:00, 2348426.92it/s]
[UPLOADED] /tmp/ssa-upload-8332/kv2_align_kv2_a_0000144.jpg (213tdFVQsg3VsMMSDHXL) [0.7s]
[UPLOADED] /tmp/ssa-upload-8332/kv1_NYUdata_NYU0428_0000103.jpg (U3cSw1b6mJYx8TQZ9ZnU) [0.8s]
Imported 8332 images from /tmp/ssa-upload-8332 to smart-scene-analyzer-2026-09-08
EXIT=0
"""


def test_maps_image_stem_to_platform_id():
    m = parse_upload_log(LOG)
    assert m == {
        "kv2_align_kv2_a_0000144": "213tdFVQsg3VsMMSDHXL",
        "kv1_NYUdata_NYU0428_0000103": "U3cSw1b6mJYx8TQZ9ZnU",
    }


def test_keys_are_stems_so_a_label_file_looks_up_directly():
    """A label file is `<stem>.txt`; the key must not carry `.jpg`."""
    m = parse_upload_log(LOG)
    assert all(not k.endswith(".jpg") for k in m)


def test_ignores_progress_bars_and_summary_lines():
    assert len(parse_upload_log(LOG)) == 2


def test_rejects_a_log_with_no_uploads():
    with pytest.raises(ValueError, match="no \\[UPLOADED\\] lines"):
        parse_upload_log("loading Roboflow project...\nEXIT=1\n")


def test_rejects_one_name_carrying_two_image_ids():
    """If the same filename went up twice, the label could land on either image.

    SUN RGB-D reuses frame filenames across sessions, so this is the failure the
    path-derived upload naming exists to prevent; the parser must not paper over it.
    """
    dup = LOG + "[UPLOADED] /tmp/x/kv2_align_kv2_a_0000144.jpg (DIFFERENTIDxxxxxxxx) [0.5s]\n"
    with pytest.raises(ValueError, match="two image IDs"):
        parse_upload_log(dup)


def test_identical_duplicate_line_is_tolerated():
    """A retried line reporting the same ID is not ambiguous."""
    same = LOG + "[UPLOADED] /tmp/x/kv2_align_kv2_a_0000144.jpg (213tdFVQsg3VsMMSDHXL) [0.5s]\n"
    assert len(parse_upload_log(same)) == 2


def test_labelmap_keys_are_string_indices_in_file_order(tmp_path):
    p = tmp_path / "classes.txt"
    p.write_text("chair\ntable\nstorage\n")
    assert build_labelmap(p) == {"0": "chair", "1": "table", "2": "storage"}


def test_labelmap_order_is_the_class_id_contract(tmp_path):
    """Reordering classes.txt must reorder the IDs, not be silently normalized."""
    p = tmp_path / "classes.txt"
    p.write_text("table\nchair\n")
    assert build_labelmap(p) == {"0": "table", "1": "chair"}


def test_rejects_an_empty_classes_file(tmp_path):
    p = tmp_path / "classes.txt"
    p.write_text("\n")
    with pytest.raises(ValueError, match="is empty"):
        build_labelmap(p)
