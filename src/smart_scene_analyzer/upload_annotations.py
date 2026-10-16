"""Attach converted YOLO labels to images already present in a Roboflow project.

This is not an upload of images — those are on the platform already and paid for. It
posts one annotation per existing image, keyed by the platform's **image ID**, which is
the only handle Roboflow accepts. Filenames are not a key.

The IDs come from the `roboflow import` log, which prints
``[UPLOADED] /path/<name>.jpg (<image_id>) [0.7s]`` per image. That log is therefore an
artifact worth keeping, not scratch output: without it the mapping has to be recovered
by paging the project's image list.

Free: no rate exists for annotation save in `roboflow:plans-and-pricing`, and no image is
re-uploaded. The alternative — Auto Label over the same 8,332 images at 1 credit/100 —
would be 83.3 credits against a 20-credit lifetime budget.

Resumable: completed image IDs are appended to a state file, so an interrupted run
continues rather than restarting. ``--overwrite`` replaces annotations already on the
platform, which is what a taxonomy change requires.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_UPLOADED = re.compile(r"^\[UPLOADED\]\s+(?P<path>\S+)\s+\((?P<image_id>[^)]+)\)")


def parse_upload_log(text: str) -> dict[str, str]:
    """Map image **stem** -> platform image ID from a `roboflow import` log.

    Keyed by stem rather than filename so a label file's ``.txt`` name looks up
    directly. Raises if one stem carries two different IDs — that would mean the upload
    put the same name on the platform twice and the labels could land on either.
    """
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        m = _UPLOADED.match(line)
        if not m:
            continue
        stem = Path(m.group("path")).stem
        image_id = m.group("image_id")
        if stem in mapping and mapping[stem] != image_id:
            raise ValueError(
                f"{stem!r} maps to two image IDs ({mapping[stem]} and {image_id}); "
                "the upload is ambiguous and annotations could land on either"
            )
        mapping[stem] = image_id
    if not mapping:
        raise ValueError("no [UPLOADED] lines found in the upload log")
    return mapping


def build_labelmap(classes_txt: Path) -> dict[str, str]:
    """YOLO labels carry integer IDs; Roboflow needs the names alongside them.

    Line order in ``classes.txt`` is the ID, which is the same contract as
    ``docs/taxonomy.md``.
    """
    names = classes_txt.read_text().split()
    if not names:
        raise ValueError(f"{classes_txt} is empty")
    return {str(i): n for i, n in enumerate(names)}


@dataclass
class Outcome:
    ok: int = 0
    already: int = 0
    no_image_id: int = 0
    failed: int = 0
    warnings: int = 0
    boxes: int = 0
    errors: Counter[str] = field(default_factory=Counter)
    warn_codes: Counter[str] = field(default_factory=Counter)

    def print(self) -> None:
        print(f"\n{'=' * 60}\nANNOTATION UPLOAD\n{'=' * 60}")
        print(f"  succeeded            {self.ok:7d}")
        print(f"  already annotated    {self.already:7d}")
        print(f"  no image ID in log   {self.no_image_id:7d}")
        print(f"  failed               {self.failed:7d}")
        print(f"  boxes posted         {self.boxes:7d}")
        if self.warn_codes:
            print(f"\n  platform warnings ({self.warnings} annotations affected):")
            for code, n in self.warn_codes.most_common():
                print(f"    {code:<32s} {n:7d}")
        if self.errors:
            print("\n  errors by type:")
            for kind, n in self.errors.most_common(10):
                print(f"    {kind:<32s} {n:7d}")


def upload(
    project_root: Path,
    *,
    dataset: Path,
    upload_log: Path,
    state_file: Path,
    project_id: str,
    job_name: str,
    workers: int = 10,
    limit: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Outcome:
    from roboflow import Roboflow  # imported here so --dry-run needs no credentials

    labels_dir = dataset / "labels"
    labelmap = build_labelmap(dataset / "classes.txt")
    id_map = parse_upload_log(upload_log.read_text())

    label_files = sorted(labels_dir.glob("*.txt"))
    done: set[str] = set()
    if state_file.exists():
        done = set(state_file.read_text().split())

    todo = [p for p in label_files if p.stem not in done]
    if limit is not None:
        todo = todo[:limit]

    print(f"project:   {project_id}")
    print(f"labels:    {len(label_files)} total, {len(done)} already done, {len(todo)} to post")
    print(f"labelmap:  {labelmap}")
    print(f"overwrite: {overwrite}")

    out = Outcome()
    if dry_run:
        missing = [p.stem for p in todo if p.stem not in id_map]
        out.no_image_id = len(missing)
        out.boxes = sum(len([ln for ln in p.read_text().splitlines() if ln.strip()]) for p in todo)
        print(f"\nDRY RUN — would post {len(todo) - len(missing)} annotations")
        if missing:
            print(f"  {len(missing)} label files have no image ID; first few: {missing[:5]}")
        return out

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(project_id)

    lock = threading.Lock()
    state_handle = state_file.open("a")

    def post(path: Path) -> tuple[Path, str, Any]:
        image_id = id_map.get(path.stem)
        if image_id is None:
            return path, "no_image_id", None
        try:
            result, _elapsed, _retries = project.save_annotation(
                annotation_path=str(path),
                annotation_labelmap=labelmap,
                image_id=image_id,
                job_name=job_name,
                annotation_overwrite=overwrite,
                num_retry_uploads=3,
            )
        except Exception as exc:
            return path, "failed", exc
        return path, "ok", result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(post, p) for p in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            path, status, payload = fut.result()
            n_boxes = len([ln for ln in path.read_text().splitlines() if ln.strip()])
            if status == "ok":
                out.ok += 1
                out.boxes += n_boxes
                if isinstance(payload, dict) and payload.get("warnings"):
                    out.warnings += 1
                    for bad in payload.get("badAnnotations") or []:
                        out.warn_codes[str(bad.get("code", "unknown"))] += 1
                with lock:
                    state_handle.write(path.stem + "\n")
                    state_handle.flush()
            elif status == "no_image_id":
                out.no_image_id += 1
            else:
                out.failed += 1
                out.errors[type(payload).__name__] += 1
                if out.failed <= 5:
                    print(f"  FAIL {path.stem}: {type(payload).__name__}: {payload}")
            if i % 500 == 0:
                print(f"  ... {i}/{len(todo)}  ok={out.ok} failed={out.failed}")

    state_handle.close()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", type=Path, default=Path("build/yolo-sunrgbd"))
    p.add_argument("--upload-log", type=Path, default=Path("/tmp/ssa-upload.log"))
    p.add_argument("--state", type=Path, default=Path("build/annotations-posted.txt"))
    p.add_argument("--project", default="smart-scene-analyzer-2026-09-08")
    p.add_argument("--job-name", default="sun-rgbd-yolo-2026-09-10")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    out = upload(
        Path.cwd(),
        dataset=a.dataset,
        upload_log=a.upload_log,
        state_file=a.state,
        project_id=a.project,
        job_name=a.job_name,
        workers=a.workers,
        limit=a.limit,
        overwrite=a.overwrite,
        dry_run=a.dry_run,
    )
    out.print()
    return 0 if out.failed == 0 and out.no_image_id == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
