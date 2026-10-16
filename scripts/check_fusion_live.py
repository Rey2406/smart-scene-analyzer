#!/usr/bin/env python3
"""Lesson 04, step 3b — detection, depth, and fusion on ONE frame from your own room.

`docs/evaluation-v1.md` measures the model on a held-out split built from the same
sensor family, the same kind of room, annotated by the same process. This script measures
nothing — it puts detection, depth, and the REAL fusion layer on one frame from your own
camera and lets you disagree with what comes out. A model that scores well on the split
and falls apart on your own desk has told you something no metric in this lesson would.

**Imports `smart_scene_analyzer.fusion.fuse` — does not reimplement the reduction.** A
verification script that reimplements the thing it verifies proves that two pieces of
code written in the same hour agree with each other, which is not the question.

## A real architectural seam this script has to cross, not paper over

`fusion.fuse` expects `DepthMap.values` to live at `DEPTH_INPUT` resolution (the model's
native output grid, e.g. 518x518), unresized, with boxes mapped into it via a letterbox
transform (`docs/architecture.md` R4 — the depth map is deliberately never resampled to
`SOURCE`). But `depth.estimate_relative_inverse_depth` (Lesson 03) resizes its output to
`SOURCE` resolution directly, for the simpler job Lesson 03 asked of it. Those two
contracts disagree about what "the depth map" is sized to.

This script bridges them honestly rather than silently: it wraps the SOURCE-resolution
array in an **identity** `LetterboxTransform` (scale=1, no padding, in==out==SOURCE size).
Boxes are already in `SOURCE` space, so `letterbox_forward` under an identity transform is
a no-op and lands them correctly in this SOURCE-sized grid. This is NOT what the real
on-device `DEPTH_INPUT`-resolution pipeline will do once the depth export exists (it
doesn't yet — `docs/model-card-v1.md` records that export as failed) — it is today's
honest wiring, using the reference implementations that actually exist.

## It writes files. It never opens a window.

`opencv-python-headless` has no GUI backend — `cv2.imshow` raises. This script hands you
two PNGs to open yourself.

Units, everywhere a number is printed: **relative inverse depth** — larger is nearer, no
unit, no scale, comparable only within this one frame. Nothing printed here is a distance.

Usage:
    uv run python scripts/check_fusion_live.py --camera 0
    uv run python scripts/check_fusion_live.py --image photo.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smart_scene_analyzer.fusion import fuse_detections_with_depth, sort_near_to_far
from smart_scene_analyzer.types import DepthMap, Detection, LetterboxTransform

DEFAULT_TRAINED_WEIGHTS = Path("runs/detect/runs/v1-baseline/weights/best.pt")
DEFAULT_COCO_WEIGHTS = "yolo11n.pt"


def grab_frame(camera: int, warmup: int = 10):
    """Capture one frame from a camera index, as an RGB uint8 array.

    The warmup matters: most webcams need several frames to finish auto-exposure and
    white balance, and the first frame is routinely near-black. A near-black frame
    produces a confident, meaningless depth map.
    """
    import cv2

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        sys.exit(
            f"FAIL   Could not open camera {camera}.\n"
            "       On macOS the terminal needs camera permission: System Settings ->\n"
            "       Privacy & Security -> Camera. If you have no webcam, use --image."
        )
    try:
        frame = None
        for _ in range(max(warmup, 1)):
            ok, frame = capture.read()
            if not ok:
                sys.exit(f"FAIL   Camera {camera} opened but returned no frame.")
    finally:
        capture.release()

    # cv2 hands back BGR. Everything downstream (ultralytics, depth.py, PIL) assumes RGB,
    # and a silent swap here produces plausible-looking boxes and a plausible-looking
    # depth map that are both subtly wrong — not an error, just wrong color-conditioned
    # predictions.
    return frame[:, :, ::-1].copy()


def load_image(path: Path):
    import numpy as np
    from PIL import Image

    if not path.exists():
        sys.exit(f"FAIL   {path} does not exist.")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resolve_weights(explicit: str | None) -> str:
    if explicit:
        return explicit
    if DEFAULT_TRAINED_WEIGHTS.is_file():
        return str(DEFAULT_TRAINED_WEIGHTS)
    return DEFAULT_COCO_WEIGHTS


def run_detection(frame, weights: str, conf: float) -> list[Detection]:
    """Build REAL `Detection` records (`types.py`) from ultralytics' output — in `SOURCE`
    space, which is what `fusion.fuse` requires."""
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(weights)
    results = model.predict(frame, conf=conf, verbose=False)
    detections: list[Detection] = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=int(box.cls[0]),
                    label=result.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                )
            )
    return detections


def run_depth(frame) -> DepthMap:
    """Wrap `depth.py`'s SOURCE-resolution output in an identity transform — see the
    module docstring for why this is a deliberate bridge, not the eventual on-device
    contract. Used here only for `render_depth`; the fusion call itself goes through
    `fuse_detections_with_depth`, which does this same wrapping internally from the raw
    array."""
    from smart_scene_analyzer.depth import estimate_relative_inverse_depth

    values = estimate_relative_inverse_depth(frame)
    h, w = values.shape
    identity = LetterboxTransform(scale=1.0, pad_x=0.0, pad_y=0.0, in_w=w, in_h=h, out_w=w, out_h=h)
    return DepthMap(values=values, transform=identity, valid=(0.0, 0.0, float(w), float(h)))


def annotate(frame, ranked, path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    n = max(len(ranked), 1)
    for fused in ranked:
        rank = fused.depth_rank
        shade = int(255 * (1 - (rank / n))) if rank is not None else 80
        box = [int(v) for v in fused.detection.bbox]
        draw.rectangle(box, outline=(255, shade, 0), width=3)
        rank_str = f"#{rank}" if rank is not None else "#-"
        caption = f"{rank_str} {fused.detection.label} {fused.detection.confidence:.2f}"
        if fused.relative_depth is not None:
            caption += f"  d={fused.relative_depth:.2f}"
        draw.text((box[0] + 4, box[1] + 4), caption, fill=(255, 255, 0))
    image.save(path)


def render_depth(depth_map: DepthMap, path: Path) -> None:
    import numpy as np
    from PIL import Image

    values = depth_map.values
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if high - low < 1e-9:
        print("  WARN   Depth map is constant — nothing to render.")
        return
    scaled = ((values - low) / (high - low) * 255).astype("uint8")
    Image.fromarray(scaled).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detection + depth + fusion on one real frame.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="Camera index, usually 0")
    source.add_argument("--image", type=Path, help="Image file, if you have no webcam")
    parser.add_argument(
        "--weights",
        default=None,
        help="Detection weights. Defaults to the "
        "trained v1-baseline checkpoint if present, else COCO yolo11n.pt.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Where to write PNGs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    weights = resolve_weights(args.weights)
    using_trained = weights == str(DEFAULT_TRAINED_WEIGHTS)
    print(f"  weights = {weights}  ({'trained v1-baseline' if using_trained else 'COCO, untuned'})")

    frame = grab_frame(args.camera) if args.camera is not None else load_image(args.image)
    print(f"  frame {frame.shape[1]}x{frame.shape[0]}")

    detections = run_detection(frame, weights, args.conf)
    depth_map = run_depth(frame)

    fused = fuse_detections_with_depth(
        detections,
        depth_map.values,
        image_height=depth_map.values.shape[0],
        image_width=depth_map.values.shape[1],
    )

    annotated_path = args.out_dir / "fusion_live_annotated.png"
    depth_path = args.out_dir / "fusion_live_depth.png"

    if not fused:
        print("\n  Zero detections. That is a successful result with an empty list, not a")
        print("  failure — the fusion layer was still exercised, there was just nothing to")
        print("  fuse.")
        annotate(frame, [], annotated_path)
        render_depth(depth_map, depth_path)
        print(f"\n  Wrote {annotated_path}")
        print(f"  Wrote {depth_path}")
        return 0

    ranked = sort_near_to_far(fused)

    print(
        f"\n  {'rank':>4}  {'class':<10} {'conf':>5}  {'rel.depth':>10}  "
        f"{'spread':>7}  {'n px':>5}  quality"
    )
    for d in ranked:
        rank_str = f"#{d.depth_rank}" if d.depth_rank is not None else "#-"
        depth_str = f"{d.relative_depth:10.3f}" if d.relative_depth is not None else "      none"
        spread_str = f"{d.depth_spread:7.3f}" if d.depth_spread is not None else "   none"
        print(
            f"  {rank_str:>4}  {d.detection.label:<10} {d.detection.confidence:>5.2f}  "
            f"{depth_str}  {spread_str}  {d.depth_sample_count:>5}  "
            f"{d.depth_quality}"
        )

    annotate(frame, ranked, annotated_path)
    render_depth(depth_map, depth_path)

    print(f"\n  Wrote {annotated_path}")
    print(f"  Wrote {depth_path}")
    print("\n  Values are relative inverse depth: no unit, no scale, this frame only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
