#!/usr/bin/env python3
"""Assemble a YOLO dataset version locally, in the shape Roboflow's export produces.

Why this exists: the Roboflow dataset version is blocked on credits, but the converted
dataset is already complete on disk. Lesson 03 trains YOLO11, which reads a directory
and a data.yaml — it does not read Roboflow. This produces that directory so training
is not gated on a platform balance.

What it is NOT: a Roboflow version. There is no immutable platform snapshot behind it,
so provenance is this repo's git history plus the converter's deterministic seed rather
than a version number on the platform. Regenerate the platform version from the same
converter when credits return; the labels are identical by construction.

Layout written (matching a Roboflow YOLOv11 export):

    data/v1/
      data.yaml
      train/images  train/labels
      valid/images  valid/labels
      test/images   test/labels

Images are symlinked, not copied — the sources are themselves symlinks into
data/SUNRGBD, so nothing is duplicated on disk and the source tree is never modified.

Splits are assigned deterministically and **spread across capture sessions**, the same
principle as the platform rebalance: SUN RGB-D is ordered by sensor and session, so a
contiguous split would put one sensor entirely in test and measure domain shift rather
than detection quality.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

RATIOS = {"train": 70, "valid": 20, "test": 10}


def session_of(stem: str) -> str:
    """`kv2_align_kv2_2014-12-18_...` -> `kv2_align_kv2`, the capture session."""
    parts = stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else stem


# 7 train / 2 valid / 1 test, interleaved rather than blocked. Interleaving matters:
# dealt as [trainx7, validx2, testx1] any run shorter than the pattern takes only the
# head of it, which silently starves valid and test.
PATTERN = ("train", "train", "valid", "train", "train", "test",
           "train", "valid", "train", "train")


def assign_splits(stems: list[str], seed: int) -> dict[str, str]:
    """Deterministically map each stem to a split, 70/20/10, spread across sessions.

    Assignment is global over a *session-ordered* list rather than per-session. Sessions
    in SUN RGB-D range from one scene to hundreds, and dealing within each group means a
    group smaller than the pattern never reaches the tail of it — which is how an earlier
    version of this function produced 84/10/5 while looking correct.

    Walking the session-ordered list and dealing round-robin gives the exact global ratio
    *and* spreads every session across all three splits, because consecutive items in
    that list belong to the same session.
    """
    import random

    by_session: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        by_session[session_of(s)].append(s)

    rng = random.Random(seed)
    ordered: list[str] = []
    for session in sorted(by_session):
        group = sorted(by_session[session])
        rng.shuffle(group)
        ordered.extend(group)

    return {stem: PATTERN[i % len(PATTERN)] for i, stem in enumerate(ordered)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", type=Path, default=Path("build/yolo-sunrgbd"))
    p.add_argument("--out", type=Path, default=Path("data/v1"))
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    src_labels, src_images = a.source / "labels", a.source / "images"
    classes = (a.source / "classes.txt").read_text().split()
    if not classes:
        print(f"FAIL  {a.source}/classes.txt is empty", file=sys.stderr)
        return 1
    if classes != sorted(classes):
        # The export's order is alphabetical and not configurable (ADR 0003). If the
        # source disagrees, every integer in every label file means something else.
        print(f"FAIL  classes.txt is not alphabetical: {classes}", file=sys.stderr)
        return 1

    stems = sorted(p.stem for p in src_labels.glob("*.txt"))
    if not stems:
        print(f"FAIL  no labels in {src_labels}", file=sys.stderr)
        return 1

    splits = assign_splits(stems, a.seed)

    for split in RATIOS:
        (a.out / split / "images").mkdir(parents=True, exist_ok=True)
        (a.out / split / "labels").mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    per_class: dict[str, Counter[int]] = {s: Counter() for s in RATIOS}
    missing_images = 0

    for stem in stems:
        split = splits[stem]
        img_src = src_images / f"{stem}.jpg"
        if not img_src.exists():
            missing_images += 1
            continue

        lbl_dst = a.out / split / "labels" / f"{stem}.txt"
        img_dst = a.out / split / "images" / f"{stem}.jpg"
        lbl_dst.write_text((src_labels / f"{stem}.txt").read_text())
        if not img_dst.exists():
            img_dst.symlink_to(img_src.resolve())

        counts[split] += 1
        for line in lbl_dst.read_text().splitlines():
            if line.strip():
                per_class[split][int(line.split()[0])] += 1

    # data.yaml — `names` order IS the class-ID contract.
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    (a.out / "data.yaml").write_text(
        f"# Generated locally by scripts/make_local_version.py (seed={a.seed}).\n"
        f"# NOT a Roboflow dataset version — see the module docstring.\n"
        f"path: {a.out.resolve()}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(classes)}\nnames:\n{names}\n"
    )

    total = sum(counts.values())
    print(f"source:  {a.source}  ({len(stems)} labelled images)")
    print(f"output:  {a.out}\n")
    print(f"{'split':8s} {'images':>8s} {'share':>8s}")
    for split in RATIOS:
        pct = 100.0 * counts[split] / total if total else 0.0
        print(f"{split:8s} {counts[split]:8d} {pct:7.2f}%")
    print(f"{'TOTAL':8s} {total:8d}")
    print(f"\nreconciles: {'YES' if total + missing_images == len(stems) else 'NO'}"
          f"   (images missing from source: {missing_images})")

    print(f"\n{'class':10s} " + "".join(f"{s:>8s}" for s in RATIOS))
    for cid, name in enumerate(classes):
        row = "".join(f"{per_class[s][cid]:8d}" for s in RATIOS)
        print(f"{cid} {name:8s}" + row)

    return 0 if total == len(stems) - missing_images and total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
