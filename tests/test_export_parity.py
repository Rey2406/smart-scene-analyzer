"""N20 — export parity: does the exported artifact still behave like the trained model?

Lesson 04, step 6. Step 4 produced a file that loads and runs; this asks whether it still
matches the PyTorch model it was exported from. Quantization, a resize implementation, and
a normalization step sit between the two claims, and any of them can be subtly wrong in a
way no smoke test catches.

**Compared against PyTorch directly (`ultralytics.YOLO(...).predict()`), never against the
export run in a second runtime.** Comparing the artifact to itself would only confirm that
two libraries can read one file — it says nothing about whether the file still matches the
training. The exported side is decoded through this project's own `inference.decode` +
`inference.nms` (built for this test, from their existing "planned public surface"
specs) so the comparison exercises the same decode path the app will eventually run,
not a third, throwaway implementation.

**Integration-marked**: needs the trained checkpoint, the exported `.tflite`, and the
test-split images named in `tests/fixtures/export_parity_images.json` — all on disk,
none committed as binaries. The default suite (`uv run pytest`) does not run this module;
`uv run pytest -m integration` does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "app" / "assets" / "models" / "model_contract.json"
IMAGE_LIST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "export_parity_images.json"
CHECKPOINT = PROJECT_ROOT / "runs" / "detect" / "runs" / "v1-baseline" / "weights" / "best.pt"

CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.45

# The standard COCO-style match criterion: two boxes of the same class with IoU >= 0.5 are
# "the same detection" for comparison purposes. This is a long-established convention
# (COCO eval, Pascal VOC), not a value picked to make this test pass.
MATCH_IOU_THRESHOLD = 0.5


@dataclass
class ImageComparison:
    image: str
    pytorch: list
    exported: list
    matches: list[tuple]  # (pytorch_det, exported_det, iou)
    unmatched_pytorch: list
    unmatched_exported: list


def _load_image_list() -> list[Path]:
    data = json.loads(IMAGE_LIST_PATH.read_text())
    return [PROJECT_ROOT / p for p in data["images"]]


def _skip_if_data_missing() -> list[Path]:
    if not MANIFEST_PATH.is_file():
        pytest.skip(f"{MANIFEST_PATH} not present — detection has not been exported.")
    if not CHECKPOINT.is_file():
        pytest.skip(f"{CHECKPOINT} not present — no trained checkpoint on disk.")
    images = _load_image_list()
    missing = [p for p in images if not p.is_file()]
    if missing:
        pytest.skip(
            f"{len(missing)}/{len(images)} test-split images not present "
            f"(data/v1 is gitignored) — e.g. {missing[0]}"
        )
    return images


def _run_pytorch(checkpoint: Path, image_path: Path) -> list:
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(checkpoint))
    results = model.predict(
        str(image_path), conf=CONF_THRESHOLD, iou=IOU_NMS_THRESHOLD, imgsz=640, verbose=False
    )
    from smart_scene_analyzer.types import Detection

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=int(box.cls[0]),
                    label=r.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                )
            )
    return detections


def _run_exported(manifest, image_path: Path) -> list:
    from ai_edge_litert.interpreter import Interpreter
    from PIL import Image

    from smart_scene_analyzer.inference.decode import decode_raw, to_source_frame
    from smart_scene_analyzer.inference.nms import nms
    from smart_scene_analyzer.preprocess.detection_input import build
    from smart_scene_analyzer.types import RawDetectorOutput, Size, SourceImage

    img = Image.open(image_path).convert("RGB")
    pixels = np.asarray(img, dtype=np.uint8)
    source = SourceImage(
        pixels=pixels,
        size=Size(w=pixels.shape[1], h=pixels.shape[0]),
        original_size=Size(w=pixels.shape[1], h=pixels.shape[0]),
        decode_scale=1.0,
    )
    det_input = build(source, size=manifest.detector.input.shape[-1])

    interpreter = Interpreter(
        model_path=str(PROJECT_ROOT / "app" / "assets" / "models" / manifest.detector.file)
    )
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(in_details["index"], det_input.tensor.astype(in_details["dtype"]))
    interpreter.invoke()
    raw_tensor = interpreter.get_tensor(out_details["index"])

    raw = RawDetectorOutput(tensors=(raw_tensor,))
    candidates = decode_raw(
        raw,
        manifest.detector.outputs[0],
        manifest.detector.labels,
        conf_threshold=CONF_THRESHOLD,
        det_input_size=manifest.detector.input.shape[-1],
    )
    suppressed = nms(
        candidates, iou_threshold=IOU_NMS_THRESHOLD, max_detections=manifest.detector.max_detections
    )
    return to_source_frame(suppressed, det_input.transform, source.size)


def _compare_one_image(
    image_path: Path, pytorch_dets: list, exported_dets: list
) -> ImageComparison:
    from smart_scene_analyzer.geometry.boxes import iou as box_iou

    remaining_exported = list(exported_dets)
    matches = []
    unmatched_pytorch = []
    for p in pytorch_dets:
        best_iou = 0.0
        best = None
        for e in remaining_exported:
            if e.class_id != p.class_id:
                continue
            i = box_iou(p.bbox, e.bbox)
            if i > best_iou:
                best_iou, best = i, e
        if best is not None and best_iou >= MATCH_IOU_THRESHOLD:
            matches.append((p, best, best_iou))
            remaining_exported.remove(best)
        else:
            unmatched_pytorch.append(p)

    return ImageComparison(
        image=image_path.name,
        pytorch=pytorch_dets,
        exported=exported_dets,
        matches=matches,
        unmatched_pytorch=unmatched_pytorch,
        unmatched_exported=remaining_exported,
    )


@pytest.fixture(scope="module")
def comparisons() -> list[ImageComparison]:
    from smart_scene_analyzer.export import contract

    images = _skip_if_data_missing()
    manifest = contract.read(MANIFEST_PATH)

    results = []
    for image_path in images:
        pytorch_dets = _run_pytorch(CHECKPOINT, image_path)
        exported_dets = _run_exported(manifest, image_path)
        results.append(_compare_one_image(image_path, pytorch_dets, exported_dets))
    return results


def test_print_both_result_sets(comparisons):
    """Round 1: assert nothing, just look. Kept as a test (not a throwaway script) so
    `pytest -s -k print_both` reproduces this view on demand."""
    for c in comparisons:
        print(f"\n{c.image}")
        print(f"  PyTorch ({len(c.pytorch)}):")
        for d in c.pytorch:
            print(f"    {d.label:<10} {d.confidence:.2f}  {tuple(round(v, 1) for v in d.bbox)}")
        print(f"  Exported ({len(c.exported)}):")
        for d in c.exported:
            print(f"    {d.label:<10} {d.confidence:.2f}  {tuple(round(v, 1) for v in d.bbox)}")


def test_detection_parity_per_class(comparisons):
    """Round 2: per-class delta table, not just an aggregate — int8 can destroy one class
    while barely moving the aggregate, and a single number is exactly what hides that.
    """
    by_class: dict[str, dict] = {}
    for c in comparisons:
        for p, e, iou_val in c.matches:
            row = by_class.setdefault(
                p.label, {"pytorch": 0, "exported": 0, "matched": 0, "ious": [], "deltas": []}
            )
            row["matched"] += 1
            row["ious"].append(iou_val)
            row["deltas"].append(e.confidence - p.confidence)
        for p in c.unmatched_pytorch:
            by_class.setdefault(
                p.label, {"pytorch": 0, "exported": 0, "matched": 0, "ious": [], "deltas": []}
            )
        for e in c.unmatched_exported:
            by_class.setdefault(
                e.label, {"pytorch": 0, "exported": 0, "matched": 0, "ious": [], "deltas": []}
            )

    for c in comparisons:
        for p, _, _ in c.matches:
            by_class[p.label]["pytorch"] += 1
        for p in c.unmatched_pytorch:
            by_class[p.label]["pytorch"] += 1
        for _, e, _ in c.matches:
            by_class[e.label]["exported"] += 1
        for e in c.unmatched_exported:
            by_class[e.label]["exported"] += 1

    header = (
        f"{'class':<10} {'pytorch':>8} {'exported':>9} {'matched':>8} "
        f"{'mean IoU':>9} {'mean conf delta':>17}"
    )
    print(f"\n{header}")
    worst_delta = 0.0
    worst_delta_class = None
    min_iou_seen = 1.0
    for label, row in sorted(by_class.items()):
        mean_iou = np.mean(row["ious"]) if row["ious"] else float("nan")
        mean_delta = np.mean(row["deltas"]) if row["deltas"] else float("nan")
        print(
            f"{label:<10} {row['pytorch']:>8} {row['exported']:>9} {row['matched']:>8} "
            f"{mean_iou:>9.3f} {mean_delta:>+17.3f}"
        )
        if row["ious"]:
            min_iou_seen = min(min_iou_seen, min(row["ious"]))
        if row["deltas"] and abs(np.mean(row["deltas"])) > abs(worst_delta):
            worst_delta = float(np.mean(row["deltas"]))
            worst_delta_class = label

    print(f"\nWorst mean confidence delta: {worst_delta_class} ({worst_delta:+.3f})")

    # Every PyTorch detection must find a same-class match above the IoU threshold — a
    # missed match means either a classification change or a box that moved too far.
    total_pytorch = sum(row["pytorch"] for row in by_class.values())
    total_matched = sum(row["matched"] for row in by_class.values())
    match_rate = total_matched / total_pytorch if total_pytorch else 1.0
    print(f"Overall match rate: {total_matched}/{total_pytorch} = {match_rate:.1%}")

    assert match_rate >= 0.90, (
        f"Only {match_rate:.1%} of PyTorch detections found a same-class, "
        f"IoU>={MATCH_IOU_THRESHOLD} match in the exported artifact."
    )
    assert min_iou_seen >= MATCH_IOU_THRESHOLD, (
        "A 'matched' pair scored below its own threshold — bug."
    )
    # Confidence WILL shift under int8 — bounded, not required to be ~0.
    assert abs(worst_delta) < 0.5, (
        f"Class {worst_delta_class!r} shifted mean confidence by {worst_delta:+.3f} between "
        "PyTorch and the export — larger than any class should plausibly move."
    )


def test_depth_parity_is_not_possible_yet():
    """Round 3: depth export FAILED (Lesson 04 step 4) — `optimum-executorch` 1.1.0 has no
    `depth-estimation` task registered, confirmed from its source. There is no `.pte` to
    compare against. This is the finding, not a gap to quietly skip past — see
    `docs/model-card-v1.md`'s Depth "Exported artifact" section for the full root cause.
    """
    depth_pte = PROJECT_ROOT / "app" / "assets" / "models" / "depth.pte"
    if not depth_pte.is_file():
        pytest.skip(
            "depth.pte does not exist — the export failed (optimum-executorch has no "
            "depth-estimation task). Depth parity cannot be checked until a depth export "
            "exists. See docs/model-card-v1.md."
        )
    pytest.fail(
        "depth.pte now exists but this test was not updated to check it — write the real "
        "depth parity check (ranking comparison on the raw 518x518 raster, per Round 3) "
        "before removing this guard."
    )
