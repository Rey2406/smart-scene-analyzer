#!/usr/bin/env python3
"""Measure N4: pairwise depth-ordering accuracy against SUN RGB-D's own depth channel.

**Ordering only, never scale.** Depth Anything V2 returns relative inverse depth with an
arbitrary scale, so no absolute error can be computed and none is attempted here. What
*can* be checked is whether the model ranks two objects in the same order the sensor did,
and that is exactly what N4 asks for.

The reference and the prediction run in opposite directions, which is the whole subtlety:

- SUN RGB-D's channel is a **range** reading — larger means further away.
- Depth Anything V2 returns **relative inverse depth** — larger means nearer.

So for a correctly behaving model the two must *disagree* in sign on every pair. A script
that forgot this would report near-zero accuracy and look like a model failure.

SUN RGB-D stores its raster bit-rotated (the toolbox does `bitshift(d,-3) |
bitshift(d,13)`); undoing that is not optional, because without it values wrap and the
ordering is destroyed rather than merely rescaled. Zero means "no reading" and is masked.
"""

from __future__ import annotations

import argparse
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smart_scene_analyzer.convert_sunrgbd import flat_name
from smart_scene_analyzer.depth import (
    DegenerateRegionError,
    estimate_relative_inverse_depth,
)

MAT = Path("data/SUNRGBD/SUNRGBDMeta2DBB_v2.mat")


def decode_sunrgbd_depth(path: Path) -> np.ndarray:
    """Undo SUN RGB-D's bit rotation. Returns float32; 0 marks an invalid reading.

    Larger values are FURTHER — the opposite sense to the model under test.
    """
    raw = np.array(Image.open(path), dtype=np.uint16)
    rotated = (raw >> np.uint16(3)) | (raw << np.uint16(13))
    out = rotated.astype(np.float32)
    out[raw == 0] = 0.0
    return out


def reference_region(ref: np.ndarray, box: tuple[float, ...], min_valid: int = 50) -> float | None:
    """Median of the valid readings inside a box, or None when too few exist."""
    h, w = ref.shape
    x1, y1, x2, y2 = box
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if cx1 >= cx2 or cy1 >= cy2:
        return None
    region = ref[cy1:cy2, cx1:cx2]
    valid = region[region > 0]
    if valid.size < min_valid:
        return None
    return float(np.median(valid))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="test")
    p.add_argument("--data", type=Path, default=Path("data/v1"))
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--min-separation", type=float, default=0.0,
                   help="skip pairs whose reference readings differ by less than this")
    a = p.parse_args(argv)

    # stem -> (rgb path, depth path), from the metadata rather than by un-flattening names
    index: dict[str, tuple[Path, Path]] = {}
    for rec in loadmat(MAT)["SUNRGBDMeta2DBB"].ravel():
        seq = str(rec["sequenceName"].ravel()[0])
        rgb = str(rec["rgbname"].ravel()[0])
        dep = str(rec["depthname"].ravel()[0])
        stem = flat_name(seq, rgb).rsplit(".", 1)[0]
        index[stem] = (Path("data") / seq / "image" / rgb,
                       Path("data") / seq / "depth_bfx" / dep)

    label_dir = a.data / a.split / "labels"
    stems = sorted(p.stem for p in label_dir.glob("*.txt"))
    random.Random(a.seed).shuffle(stems)

    pairs_total = pairs_correct = 0
    images_used = images_skipped = 0

    for stem in stems:
        if images_used >= a.limit:
            break
        if stem not in index:
            images_skipped += 1
            continue
        rgb_path, depth_path = index[stem]
        if not (rgb_path.is_file() and depth_path.is_file()):
            images_skipped += 1
            continue

        lines = (label_dir / f"{stem}.txt").read_text().splitlines()
        rows = [r.split() for r in lines if r.strip()]
        if len(rows) < 2:
            continue

        with Image.open(rgb_path) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            predicted = estimate_relative_inverse_depth(rgb, device=a.device)
        reference = decode_sunrgbd_depth(depth_path)
        if reference.shape != predicted.shape:
            images_skipped += 1
            continue

        objects = []
        for r in rows:
            cx, cy, nw, nh = (float(v) for v in r[1:])
            box = ((cx - nw / 2) * width, (cy - nh / 2) * height,
                   (cx + nw / 2) * width, (cy + nh / 2) * height)
            ref = reference_region(reference, box)
            if ref is None:
                continue
            try:
                pred = float(np.median(predicted[
                    max(0, int(box[1])):int(np.ceil(box[3])),
                    max(0, int(box[0])):int(np.ceil(box[2]))
                ]))
            except (DegenerateRegionError, ValueError):
                continue
            objects.append((ref, pred))

        if len(objects) < 2:
            continue
        images_used += 1

        for (ref_a, pred_a), (ref_b, pred_b) in combinations(objects, 2):
            if abs(ref_a - ref_b) <= a.min_separation:
                continue
            pairs_total += 1
            # reference: smaller reading = nearer. prediction: larger value = nearer.
            if (ref_a < ref_b) == (pred_a > pred_b):
                pairs_correct += 1

    acc = pairs_correct / pairs_total if pairs_total else 0.0
    print(f"\n{'=' * 64}")
    print("N4 — pairwise depth-ordering accuracy (ordering only, no scale)")
    print(f"{'=' * 64}")
    print(f"  split                 {a.split} of {a.data}")
    print(f"  images measured       {images_used}  (skipped {images_skipped})")
    print(f"  object pairs          {pairs_total}")
    print(f"  correctly ordered     {pairs_correct}")
    print(f"  accuracy              {acc:.4f}")
    print(f"  N4 target             >= 0.90   -> {'MET' if acc >= 0.90 else 'NOT MET'}")
    print(f"{'=' * 64}")
    return 0 if acc >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
