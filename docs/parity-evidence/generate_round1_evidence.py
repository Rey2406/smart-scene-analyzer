"""Regenerates docs/parity-evidence/{python-reference,app-result}.json for Lesson 05
step 12's parity report. See docs/parity-report.md and round1-ondevice-patch-test.md for
what this evidence means -- this script only reproduces how it was produced.

Requires app-depth-raw.json already present (the real on-device depth.pte dump) -- this
script does not talk to a device; it only runs the PyTorch/HF references and reduces both
depth maps through the project's own fuse().
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from smart_scene_analyzer.export.executorch import DEPTH_SIZE, _load_depth_wrapper
from smart_scene_analyzer.fusion import fuse
from smart_scene_analyzer.geometry.letterbox import content_rect, make_transform
from smart_scene_analyzer.types import DepthMap, Detection, LetterboxTransform, Size

PROJECT_ROOT = Path(".")
TEST_IMAGES = PROJECT_ROOT / "app/assets/test-images"
IMAGE_PATH = TEST_IMAGES / "kv2_align_kv2_2014-12-18_11-47-06_260595134347_0000037.jpg"
CHECKPOINT = PROJECT_ROOT / "runs/detect/runs/v1-baseline/weights/best.pt"
DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"
EVIDENCE_DIR = PROJECT_ROOT / "docs/parity-evidence"
APP_DEPTH_DUMP = EVIDENCE_DIR / "app-depth-raw.json"

CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.45
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_square_np(image: Image.Image, size: int) -> np.ndarray:
    w, h = image.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


def detection_record(f, source: str) -> dict:
    return {
        "class_id": f.detection.class_id,
        "label": f.detection.label,
        "confidence": f.detection.confidence,
        "box_source_px": list(f.detection.bbox),
        "relative_depth": f.relative_depth,
        "depth_rank": f.depth_rank,
        "depth_spread": f.depth_spread,
        "depth_quality": f.depth_quality,
        "depth_sample_count": f.depth_sample_count,
        "depth_source": source,
    }


def main() -> None:
    img = Image.open(IMAGE_PATH).convert("RGB")
    orig_w, orig_h = img.size
    print(f"image: {IMAGE_PATH.name} {orig_w}x{orig_h}")

    # --- 1. PyTorch reference detections (ORIGINAL pixel space, treated as SOURCE here) ---
    yolo = YOLO(str(CHECKPOINT))
    results = yolo.predict(
        str(IMAGE_PATH), conf=CONF_THRESHOLD, iou=IOU_NMS_THRESHOLD, imgsz=640, verbose=False
    )
    detections: list[Detection] = []
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
    print(f"PyTorch detections: {len(detections)}")
    for d in detections:
        print(f"  {d.label:<10} conf={d.confidence:.3f} box={tuple(round(v, 1) for v in d.bbox)}")

    # --- 2. PyTorch reference depth (eager, same project loading pattern) ---
    wrapper = _load_depth_wrapper(DEPTH_CHECKPOINT)
    tensor_in = letterbox_square_np(img, DEPTH_SIZE)
    with torch.no_grad():
        depth_out = wrapper(torch.from_numpy(tensor_in).contiguous()).numpy()
    depth_out = np.asarray(depth_out).squeeze()
    print(f"PyTorch eager depth output shape: {depth_out.shape}, dtype {depth_out.dtype}")
    print(f"  min={depth_out.min():.4f} max={depth_out.max():.4f} mean={depth_out.mean():.4f}")

    # Resize to [DEPTH_SIZE, DEPTH_SIZE] if the raw model output isn't already that shape
    # (DepthAnything's predicted_depth is typically already [H,W] at input resolution, but
    # verified here rather than assumed).
    if depth_out.shape != (DEPTH_SIZE, DEPTH_SIZE):
        depth_out = np.array(
            Image.fromarray(depth_out).resize((DEPTH_SIZE, DEPTH_SIZE), Image.Resampling.BILINEAR)
        )
        print(f"  resized to {depth_out.shape}")

    py_transform = make_transform(Size(w=orig_w, h=orig_h), DEPTH_SIZE, DEPTH_SIZE)
    print(
        f"Python depth transform: scale={py_transform.scale:.6f} "
        f"pad=({py_transform.pad_x:.2f},{py_transform.pad_y:.2f})"
    )

    py_depth_map = DepthMap(
        values=depth_out.astype(np.float32),
        transform=py_transform,
        valid=content_rect(py_transform),
    )

    # --- 3. App's REAL on-device depth map (dumped from the simulator) ---
    app_dump = json.loads(APP_DEPTH_DUMP.read_text())
    app_depth_values = np.array(app_dump["data"], dtype=np.float32).reshape(
        app_dump["depthHeight"], app_dump["depthWidth"]
    )
    s1 = app_dump["sourceSize"]["width"] / app_dump["originalSize"]["width"]
    combined_scale = s1 * app_dump["scale"]
    combined_pad_x = app_dump["padX"]
    combined_pad_y = app_dump["padY"]
    app_transform = LetterboxTransform(
        scale=combined_scale,
        pad_x=combined_pad_x,
        pad_y=combined_pad_y,
        in_w=orig_w,
        in_h=orig_h,
        out_w=app_dump["depthWidth"],
        out_h=app_dump["depthHeight"],
    )
    print(
        f"App depth transform (combined, original->depth_input): "
        f"scale={combined_scale:.6f} pad=({combined_pad_x:.2f},{combined_pad_y:.2f})"
    )
    print(
        f"App depth stats (reported on-device): "
        f"min={app_dump['min']:.4f} max={app_dump['max']:.4f} mean={app_dump['mean']:.4f}"
    )

    app_depth_map = DepthMap(
        values=app_depth_values,
        transform=app_transform,
        valid=content_rect(app_transform),
    )

    # --- 4. fuse() -- the project's OWN reference reduction -- for both depth maps, SAME
    #        detections ---
    fused_python = fuse(detections, py_depth_map)
    fused_app = fuse(detections, app_depth_map)

    print("\n--- Fused (Python reference depth) ---")
    for f in fused_python:
        print(f"  {f.detection.label:<10} rank={f.depth_rank} median={f.relative_depth}")

    print("\n--- Fused (App on-device depth) ---")
    for f in fused_app:
        print(f"  {f.detection.label:<10} rank={f.depth_rank} median={f.relative_depth}")

    # --- 5. Emit JSON evidence ---
    out_python = {
        "image": IMAGE_PATH.name,
        "image_size": {"width": orig_w, "height": orig_h},
        "detection_source": (
            "PyTorch (ultralytics.YOLO on runs/detect/runs/v1-baseline/weights/best.pt)"
        ),
        "depth_source": (
            "PyTorch eager (transformers.AutoModelForDepthEstimation, "
            "depth-anything/Depth-Anything-V2-Small-hf)"
        ),
        "detections": [detection_record(f, "python_eager") for f in fused_python],
    }
    out_app = {
        "image": IMAGE_PATH.name,
        "image_size": {"width": orig_w, "height": orig_h},
        "detection_source": (
            "NOT on-device -- react-native-fast-tflite Invoke() fails on detector.tflite "
            "(confirmed again this round, see round1-ondevice-patch-test.md); boxes below "
            "are BORROWED from the PyTorch reference set so depth ORDERING can still be "
            "checked against something real. These are NOT the app's own detections."
        ),
        "depth_source": (
            "REAL on-device react-native-executorch execute() on depth.pte, "
            "iPhone 17 Simulator iOS 27.0, image docs/parity-evidence/app-depth-raw.json"
        ),
        "detections": [detection_record(f, "app_on_device") for f in fused_app],
    }

    (EVIDENCE_DIR / "python-reference.json").write_text(json.dumps(out_python, indent=2))
    (EVIDENCE_DIR / "app-result.json").write_text(json.dumps(out_app, indent=2))
    print("\nWrote docs/parity-evidence/python-reference.json and app-result.json")

    # --- 6. Depth ORDERING comparison ---
    print("\n--- Rank comparison (0 = nearest) ---")
    mismatches = 0
    for i, (fp, fa) in enumerate(zip(fused_python, fused_app, strict=True)):
        match = fp.depth_rank == fa.depth_rank
        if not match:
            mismatches += 1
        print(
            f"  [{i}] {fp.detection.label:<10} python_rank={fp.depth_rank} "
            f"app_rank={fa.depth_rank} {'OK' if match else 'MISMATCH'}"
        )
    print(f"\n{len(fused_python) - mismatches}/{len(fused_python)} ranks match exactly")


if __name__ == "__main__":
    main()
