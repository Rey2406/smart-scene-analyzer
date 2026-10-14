"""Convert SUN RGB-D 2D bounding-box metadata to YOLO labels.

Source of truth for the class mapping is ``docs/taxonomy.md``. This module parses that
file rather than embedding a copy, because the taxonomy's row order *is* the class-ID
contract and two copies of it would eventually disagree.

Source geometry, established by inspecting the file (see ADR 0002):

- boxes live in ``SUNRGBDMeta2DBB[i]['groundtruth2DBB'][j]['gtBb2D']``
- four values, ``xywh``, absolute pixels, origin top-left
- class name is a string in ``['classname']``, matched **exactly** (never substring:
  ``dresser_mirror`` is a mirror, ``sofa_chair`` is a chair)

Output layout::

    <out>/images/<name>.jpg     symlink to the source image, never a copy
    <out>/labels/<name>.txt     one line per object: <class_id> <cx> <cy> <w> <h>
    <out>/classes.txt           one name per line; line order defines the integer IDs

All four coordinates are normalized to [0, 1] against the image's own pixel dimensions,
read from the file on disk rather than from the metadata.

``<name>`` reproduces the flattened naming already used for the Roboflow upload —
``<sensor>_<subdir>_<session>_<frame>.jpg`` — so a label file lines up with the image
of the same name on the platform. SUN RGB-D reuses frame filenames across sessions
(8,332 files, 3,417 distinct basenames), so the session path is not optional.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.io import loadmat

MAT_RELATIVE = Path("data/SUNRGBD/SUNRGBDMeta2DBB_v2.mat")
TAXONOMY_RELATIVE = Path("docs/taxonomy.md")
DATA_ROOT_RELATIVE = Path("data")

Records = list[tuple[int, Any]]
"""``(original .mat index, record)`` pairs. The record is a numpy void with named
fields; there is no useful static type for it."""


# --------------------------------------------------------------------------- taxonomy


@dataclass
class Taxonomy:
    """The ordered class list and the exact-match mapping onto it."""

    classes: list[str]
    """Class names in ID order. Index is the integer class ID."""

    source_to_id: dict[str, int]
    """Source label string -> class ID. Exact-match keys only."""

    def id_of(self, source_label: str) -> int | None:
        return self.source_to_id.get(source_label)


_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(.+?)\s*\|")


def parse_taxonomy(path: Path) -> Taxonomy:
    """Parse the class table out of docs/taxonomy.md.

    Expects rows shaped ``| <id> | `<class>` | `<src>`, `<src>` | ...``. Raises if the
    IDs are not 0..K-1 in order, because a gap or a swap there is the one error in this
    file that would silently mislabel every downstream artifact.
    """
    rows: list[tuple[int, str, list[str]]] = []
    for line in path.read_text().splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        cid, cls, sources_cell = int(m.group(1)), m.group(2), m.group(3)
        sources = re.findall(r"`([^`]+)`", sources_cell)
        if not sources:
            continue
        rows.append((cid, cls, sources))

    if not rows:
        raise ValueError(f"no class rows parsed from {path}")

    expected = list(range(len(rows)))
    if [r[0] for r in rows] != expected:
        raise ValueError(
            f"class IDs in {path} are {[r[0] for r in rows]}, expected {expected}. "
            "IDs must be contiguous and in order — they are the export contract."
        )

    classes = [r[1] for r in rows]
    mapping: dict[str, int] = {}
    for cid, _cls, sources in rows:
        for s in sources:
            if s in mapping and mapping[s] != cid:
                raise ValueError(
                    f"source label {s!r} is mapped to two classes "
                    f"({classes[mapping[s]]!r} and {classes[cid]!r}) in {path}"
                )
            mapping[s] = cid
    return Taxonomy(classes=classes, source_to_id=mapping)


# ------------------------------------------------------------------------ conversion


@dataclass
class Report:
    """Counts that must reconcile. A silent drop is the failure mode this guards."""

    records_total: int = 0
    records_considered: int = 0
    records_no_local_image: int = 0
    records_zero_objects: int = 0
    records_all_objects_unmapped: int = 0
    records_written: int = 0
    records_already_present: int = 0

    objects_total: int = 0
    objects_written: int = 0
    objects_unmapped: int = 0
    objects_skipped: Counter[str] = field(default_factory=Counter)

    unmapped_labels: Counter[str] = field(default_factory=Counter)
    per_class: Counter[int] = field(default_factory=Counter)
    sessions: Counter[str] = field(default_factory=Counter)
    box_area_min: float = 1.0
    box_area_max: float = 0.0

    def reconciles(self) -> bool:
        accounted = (
            self.objects_written + self.objects_unmapped + sum(self.objects_skipped.values())
        )
        return accounted == self.objects_total

    def print(self, *, dry_run: bool) -> None:
        mode = "DRY RUN — nothing written" if dry_run else "WROTE OUTPUT"
        print(f"\n{'=' * 68}\n{mode}\n{'=' * 68}")
        print("\n-- records --")
        print(f"  present in .mat                 {self.records_total:8d}")
        print(f"  considered this run             {self.records_considered:8d}")
        print(f"  skipped, image not on disk      {self.records_no_local_image:8d}")
        print(f"  skipped, zero annotated objects {self.records_zero_objects:8d}")
        print(f"  skipped, no object mapped       {self.records_all_objects_unmapped:8d}")
        print(f"  already converted (resumed)     {self.records_already_present:8d}")
        print(f"  written                         {self.records_written:8d}")

        print("\n-- objects --")
        print(f"  annotated on convertible images {self.objects_total:8d}")
        print(f"  written to labels               {self.objects_written:8d}")
        print(f"  dropped, label not in taxonomy  {self.objects_unmapped:8d}")
        for reason, n in self.objects_skipped.most_common():
            print(f"  dropped, {reason:<30s} {n:8d}")
        ok = "RECONCILES" if self.reconciles() else "DOES NOT RECONCILE"
        print(f"  -> {ok}")

        if self.per_class:
            print("\n-- per class --")
            for cid, name in sorted(self.per_class_names.items()):
                print(f"  {cid}  {name:<10s} {self.per_class[cid]:8d}")

        if self.box_area_max > 0.0:
            print("\n-- box area (fraction of image) --")
            print(f"  smallest {self.box_area_min:.6f}   largest {self.box_area_max:.6f}")

        if self.sessions:
            print(f"\n-- capture sessions represented: {len(self.sessions)} --")
            for name, n in self.sessions.most_common():
                print(f"  {n:6d}  {name}")

        if self.unmapped_labels:
            print(f"\n-- top unmapped source labels ({len(self.unmapped_labels)} distinct) --")
            for name, n in self.unmapped_labels.most_common(15):
                print(f"  {n:8d}  {name}")

    per_class_names: dict[int, str] = field(default_factory=dict)


class BoxRejectedError(ValueError):
    """A box could not be converted. ``reason`` is the itemized skip category."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def to_yolo_box(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert one SUN RGB-D box to YOLO normalized centre form.

    Args:
        x, y, w, h: source box in **absolute pixels, xywh, origin top-left** — the
            convention proven for ``gtBb2D`` in ADR 0002.
        width, height: pixel dimensions of the image the box belongs to.

    Returns:
        ``(cx, cy, nw, nh)``, all normalized to [0, 1] against ``width``/``height``,
        with the box clipped to the frame first.

    Raises:
        BoxRejectedError: the box has non-positive extent, falls entirely outside the frame,
            or lands outside [0, 1] after normalizing. Callers itemize ``reason``
            rather than dropping it silently.
    """
    if width <= 0 or height <= 0:
        raise BoxRejectedError("image has non-positive dimensions")
    if w <= 0 or h <= 0:
        raise BoxRejectedError("non-positive width/height")

    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 <= x0 or y1 <= y0:
        raise BoxRejectedError("box entirely outside image")

    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    nw = (x1 - x0) / width
    nh = (y1 - y0) / height

    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise BoxRejectedError("centre outside [0,1]")
    if not (0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
        raise BoxRejectedError("size outside (0,1]")
    return cx, cy, nw, nh


LABEL_DECIMALS = 6


def format_yolo_line(cid: int, cx: float, cy: float, nw: float, nh: float) -> str:
    """Render one label line, rounded so the box still fits inside the frame.

    Rounding a box that sits exactly on an image edge to a fixed number of decimals can
    push ``cx + nw/2`` a hair past 1.0 — by 5e-7, or 0.0003 px on a 640-wide image.
    Roboflow answers such an annotation with an ``annotation:trimmed`` warning, and on
    this dataset that is 3,402 of 51,019 boxes: harmless individually, but enough noise
    to hide a real failure in an upload log.

    So the size is floored rather than rounded, and capped to what the rounded centre
    allows. The box shrinks by at most one unit in the last decimal place.
    """
    q = 10**LABEL_DECIMALS
    ulp = 1.0 / q
    # A box must keep a non-zero extent, so its centre must sit at least half an ulp
    # from either border. Rounding alone does not guarantee that — `cx = 5e-7` rounds
    # to exactly 0.0 — so the centre is clamped to the nearest representable value that
    # leaves room. This only ever moves slivers already within one ulp of the edge.
    rcx = min(max(round(cx * q) / q, ulp), 1.0 - ulp)
    rcy = min(max(round(cy * q) / q, ulp), 1.0 - ulp)
    # Largest extent that keeps both edges inside [0, 1] given the rounded centre.
    max_w = 2.0 * min(rcx, 1.0 - rcx)
    max_h = 2.0 * min(rcy, 1.0 - rcy)
    rnw = max(int(min(nw, max_w) * q) / q, ulp)
    rnh = max(int(min(nh, max_h) * q) / q, ulp)
    return f"{cid} {rcx:.6f} {rcy:.6f} {rnw:.6f} {rnh:.6f}"


def flat_name(sequence_name: str, rgb_name: str) -> str:
    """Reproduce the flattened upload filename for a record.

    ``SUNRGBD/kv2/align_kv2/<session>`` + ``0000144.jpg``
    -> ``kv2_align_kv2_<session>_0000144.jpg``
    """
    seq = sequence_name.replace("SUNRGBD/", "", 1)
    return f"{seq.replace('/', '_')}_{rgb_name}"


def _as_str(v: object) -> str:
    if isinstance(v, np.ndarray):
        return str(v.ravel()[0]) if v.size else ""
    return str(v)


def _select_sample(records: Records, n: int, seed: int) -> Records:
    """Pick n records spread across capture sessions, deterministically.

    Taking the first n would take them all from one sensor and one afternoon, which is
    the least useful possible sample for eyeballing box geometry.
    """
    by_group: dict[str, Records] = defaultdict(list)
    for idx, rec in records:
        seq = _as_str(rec["sequenceName"])
        parts = seq.split("/")
        group = "/".join(parts[1:3]) if len(parts) >= 3 else seq
        by_group[group].append((idx, rec))

    rng = np.random.default_rng(seed)
    for g in by_group:
        by_group[g].sort(key=lambda t: t[0])
        order = rng.permutation(len(by_group[g]))
        by_group[g] = [by_group[g][i] for i in order]

    out: Records = []
    groups = sorted(by_group)
    while len(out) < n and any(by_group[g] for g in groups):
        for g in groups:
            if by_group[g] and len(out) < n:
                out.append(by_group[g].pop())
    return out


def stems_in(dataset_dir: Path) -> set[str]:
    """Image stems already converted into ``dataset_dir``.

    Used to carve a second, **disjoint** set out of the same source — the audit set must
    share no image with the set that trains the model, or it is not a control group.
    """
    labels = dataset_dir / "labels"
    if not labels.is_dir():
        return set()
    return {p.stem for p in labels.glob("*.txt")}


def convert(
    project_root: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    exclude_dir: Path | None = None,
) -> Report:
    taxonomy = parse_taxonomy(project_root / TAXONOMY_RELATIVE)
    mat_path = project_root / MAT_RELATIVE
    data_root = project_root / DATA_ROOT_RELATIVE

    print(f"taxonomy: K={len(taxonomy.classes)} {taxonomy.classes}")
    print(f"source:   {mat_path}")
    print(f"output:   {out_dir}{' (dry run)' if dry_run else ''}")

    records = list(enumerate(loadmat(mat_path)["SUNRGBDMeta2DBB"].ravel()))
    rep = Report(records_total=len(records))
    rep.per_class_names = dict(enumerate(taxonomy.classes))

    if exclude_dir is not None:
        excluded = stems_in(project_root / exclude_dir)
        before = len(records)
        records = [
            (i, r)
            for i, r in records
            if flat_name(_as_str(r["sequenceName"]), _as_str(r["rgbname"])).rsplit(".", 1)[0]
            not in excluded
        ]
        print(
            f"exclude:  {before - len(records)} records already in {exclude_dir} "
            f"({len(excluded)} stems); {len(records)} remain eligible"
        )

    if sample is not None:
        records = _select_sample(records, sample, seed)
        print(f"sample:   {len(records)} records, spread across sessions, seed={seed}")
        rep.sessions = Counter(
            "/".join(_as_str(r["sequenceName"]).split("/")[1:3]) for _i, r in records
        )
    elif limit is not None:
        records = records[:limit]
        print(f"limit:    first {len(records)} records")
    rep.records_considered = len(records)

    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "classes.txt").write_text("\n".join(taxonomy.classes) + "\n")

    for _idx, rec in records:
        seq = _as_str(rec["sequenceName"])
        rgb_name = _as_str(rec["rgbname"])
        src_img = data_root / seq / "image" / rgb_name

        if not src_img.exists():
            rep.records_no_local_image += 1
            continue

        gt = rec["groundtruth2DBB"]
        objects = list(gt.ravel()) if isinstance(gt, np.ndarray) and gt.size else []
        rep.objects_total += len(objects)
        if not objects:
            # An image with no annotated object is not evidence of an empty scene in
            # SUN RGB-D — it is an unlabelled record. Writing an empty label file would
            # assert "nothing here", which would train the detector to suppress real
            # objects. Skip the image instead.
            rep.records_zero_objects += 1
            continue

        name = flat_name(seq, rgb_name)
        stem = name.rsplit(".", 1)[0]
        out_lbl = lbl_dir / f"{stem}.txt"
        if out_lbl.exists() and not dry_run:
            rep.records_already_present += 1
            continue

        try:
            with Image.open(src_img) as im:
                width, height = im.size
        except Exception as exc:
            rep.objects_skipped[f"image unreadable ({type(exc).__name__})"] += len(objects)
            continue

        lines: list[str] = []
        for obj in objects:
            label = _as_str(obj["classname"])
            cid = taxonomy.id_of(label)
            if cid is None:
                rep.objects_unmapped += 1
                rep.unmapped_labels[label] += 1
                continue

            box = np.asarray(obj["gtBb2D"]).ravel().astype(float)
            if box.size != 4:
                rep.objects_skipped["box not 4 values"] += 1
                continue
            x, y, w, h = box

            try:
                cx, cy, nw, nh = to_yolo_box(x, y, w, h, width, height)
            except BoxRejectedError as rejected:
                rep.objects_skipped[rejected.reason] += 1
                continue

            lines.append(format_yolo_line(cid, cx, cy, nw, nh))
            area = nw * nh
            rep.box_area_min = min(rep.box_area_min, area)
            rep.box_area_max = max(rep.box_area_max, area)
            rep.objects_written += 1
            rep.per_class[cid] += 1

        if not lines:
            rep.records_all_objects_unmapped += 1
            continue

        if not dry_run:
            out_lbl.write_text("\n".join(lines) + "\n")
            link = img_dir / name
            if not link.exists():
                link.symlink_to(src_img.resolve())
        rep.records_written += 1

    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path("build/yolo-sunrgbd"))
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    p.add_argument("--limit", type=int, help="first N records, for fast iteration")
    p.add_argument("--sample", type=int, help="N records spread across capture sessions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exclude-dir",
        type=Path,
        help="skip records already converted into this dataset, so the two sets are disjoint",
    )
    a = p.parse_args(argv)

    rep = convert(
        a.project_root.resolve(),
        a.out.resolve(),
        dry_run=a.dry_run,
        limit=a.limit,
        sample=a.sample,
        seed=a.seed,
        exclude_dir=a.exclude_dir,
    )
    rep.print(dry_run=a.dry_run)
    return 0 if rep.reconciles() else 1


if __name__ == "__main__":
    sys.exit(main())
