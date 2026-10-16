"""Draw converted YOLO labels back onto the pixels, as a contact sheet.

Round 3 of the annotation-conversion procedure. Every assertion in the converter passes
on a dataset whose boxes are uniformly shifted, flipped in y, or off by one origin — the
only check that catches those is looking at them.

Reads ``<out>/labels/*.txt`` and ``<out>/classes.txt`` produced by ``convert_sunrgbd``
and writes a single grid image so the sample can be inspected at once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
]


def draw_one(img_path: Path, lbl_path: Path, classes: list[str], cell: int) -> Image.Image:
    """Render one image with its boxes, scaled into a square cell."""
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)

    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        cx, cy, nw, nh = (float(v) for v in parts[1:])
        # normalized centre form -> absolute xyxy in SOURCE image pixels
        x0 = (cx - nw / 2.0) * w
        y0 = (cy - nh / 2.0) * h
        x1 = (cx + nw / 2.0) * w
        y1 = (cy + nh / 2.0) * h
        color = PALETTE[cid % len(PALETTE)]
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        name = classes[cid] if cid < len(classes) else str(cid)
        tw = 7 * len(name) + 6
        d.rectangle([x0, max(0, y0 - 14), x0 + tw, max(14, y0)], fill=color)
        d.text((x0 + 3, max(0, y0 - 13)), name, fill=(0, 0, 0))

    im.thumbnail((cell, cell))
    canvas = Image.new("RGB", (cell, cell), (24, 24, 24))
    canvas.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
    return canvas


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("build/yolo-sunrgbd"))
    p.add_argument("--dest", type=Path, default=Path("build/preview.jpg"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--cell", type=int, default=380)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)

    classes = (a.out / "classes.txt").read_text().split()
    labels = sorted((a.out / "labels").glob("*.txt"))
    if not labels:
        raise SystemExit(f"no labels in {a.out / 'labels'}")

    rng = np.random.default_rng(a.seed)
    pick = [labels[i] for i in rng.permutation(len(labels))[: a.n]]

    cells = []
    for lbl in pick:
        img = a.out / "images" / f"{lbl.stem}.jpg"
        if not img.exists():
            continue
        cells.append((draw_one(img, lbl, classes, a.cell), lbl.stem))

    cols = a.cols
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * a.cell, rows * a.cell), (12, 12, 12))
    for i, (c, _stem) in enumerate(cells):
        sheet.paste(c, ((i % cols) * a.cell, (i // cols) * a.cell))
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.dest, quality=88)

    print(f"wrote {a.dest}  ({len(cells)} images, {cols}x{rows} grid)")
    print("classes:", classes)
    for _c, stem in cells[:5]:
        print("  sampled:", stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
