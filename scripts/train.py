#!/usr/bin/env python3
"""Fine-tune YOLO11 on an exported dataset, with the MLflow run as the unit of work.

Credits: zero. This runs on the local machine, which is why
`.claude/agents/ml-engineer.md` makes it the default path and hosted training a budgeted
comparison rather than the baseline.

The run record is the deliverable as much as the weights are. In three weeks there will
be a `.pt` file and a question about what produced it, and the answer is not in the
weights and not in the filename somebody renamed. So every argument, the resolved device,
the dataset identity, the class list, per-split counts, per-epoch metrics, the git SHA
and the plots all go into MLflow — and a run that cannot be reconstructed from that
record did not happen.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------- provenance helpers


def resolve_device(requested: str | None) -> str:
    """Pick cuda / mps / cpu, or honour an explicit override.

    Worth being loud about: a silent fall back to CPU on a machine with an accelerator
    turns a 20-minute run into a 3-hour one, and nothing errors.
    """
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_sha(root: Path) -> tuple[str, bool]:
    """(short SHA, dirty). A SHA that does not describe the code that ran is worse than none."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10
        ).stdout.strip()

    try:
        sha = run("rev-parse", "--short", "HEAD") or "unknown"
        return (sha, bool(run("status", "--porcelain")))
    except Exception:
        return ("unknown", False)


def dataset_identity(data_yaml: Path) -> dict[str, str]:
    """Identify the dataset from its path, and say honestly what backs it.

    `data/v1` normally means Roboflow dataset version 1 — an immutable platform snapshot.
    Here it is a local build (scripts/make_local_version.py, fixed seed), because the
    workspace ran out of credits before a version could be generated. Logging the folder
    name without that caveat would assert a provenance the project does not have.
    """
    version_dir = data_yaml.parent.name
    version = version_dir[1:] if version_dir.startswith("v") and version_dir[1:].isdigit() else ""
    text = data_yaml.read_text()
    local_build = "NOT a Roboflow dataset version" in text
    return {
        "dataset_dir": version_dir,
        "dataset_version": version or "unknown",
        "dataset_provenance": (
            "LOCAL build (scripts/make_local_version.py, seed=1) — NOT an immutable "
            "Roboflow dataset version; no platform snapshot stands behind it"
            if local_build
            else "Roboflow dataset version"
        ),
    }


def read_classes(data_yaml: Path) -> list[str]:
    """Class names in ID order, straight out of data.yaml's `names:` block."""
    names: dict[int, str] = {}
    in_names = False
    for line in data_yaml.read_text().splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            stripped = line.strip()
            if not stripped or not line.startswith((" ", "\t")):
                break
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                if k.strip().isdigit():
                    names[int(k)] = v.strip()
    return [names[i] for i in sorted(names)]


def split_counts(root: Path) -> dict[str, int]:
    return {
        f"images_{s}": len(list((root / s / "images").glob("*.jpg")))
        for s in ("train", "valid", "test")
        if (root / s / "images").is_dir()
    }


def check_gitignored(root: Path, project_dir: str) -> list[str]:
    """Weights and run directories are build output, never source."""
    problems = []
    for probe in (f"{project_dir.rstrip('/')}/x/weights/best.pt", "x.pt"):
        r = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", probe], capture_output=True
        )
        if r.returncode != 0:
            problems.append(probe)
    return problems


# ------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True, help="path to data.yaml")
    p.add_argument("--model", default="yolo11n.pt", help="base checkpoint")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="cuda / mps / cpu; auto-detected if omitted")
    p.add_argument("--name", required=True, help="MLflow run name")
    p.add_argument("--project", default="runs/", help="output directory for weights")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)

    root = Path.cwd()

    # Fail loudly rather than logging somewhere nobody will look. A run silently written
    # to ./mlruns is a run that will not be found when it matters.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print(
            "FAIL  MLFLOW_TRACKING_URI is not set.\n"
            "      This script will not silently log to ./mlruns — a run you cannot find\n"
            "      is not a record. Start a server and export the variable:\n\n"
            "        uv run mlflow server --host 127.0.0.1 --port 5000 \\\n"
            "          --backend-store-uri sqlite:///mlflow.db &\n"
            "        export MLFLOW_TRACKING_URI=http://127.0.0.1:5000\n",
            file=sys.stderr,
        )
        return 2

    if not a.data.is_file():
        print(f"FAIL  {a.data} not found", file=sys.stderr)
        return 1

    ignore_problems = check_gitignored(root, a.project)
    if ignore_problems:
        print(
            f"FAIL  these would be committed: {ignore_problems}\n"
            f"      Add '{a.project.rstrip('/')}/' and '*.pt' to .gitignore first.",
            file=sys.stderr,
        )
        return 1

    device = resolve_device(a.device)
    sha, dirty = git_sha(root)
    classes = read_classes(a.data)
    identity = dataset_identity(a.data)

    params = {
        "data": str(a.data.resolve()),
        "model": a.model,
        "epochs": a.epochs,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "device_requested": a.device or "auto",
        "device_resolved": device,
        "project": a.project,
        "workers": a.workers,
        "num_classes": len(classes),
        "classes": ", ".join(classes),
        **identity,
        **split_counts(a.data.parent),
    }

    print(f"{'=' * 68}\nRUN  {a.name}\n{'=' * 68}")
    for k, v in params.items():
        print(f"  {k:24s} {v}")
    print(f"  {'git_sha':24s} {sha}{' (DIRTY)' if dirty else ''}")
    print(f"  {'mlflow':24s} {tracking_uri}\n")

    import mlflow
    from ultralytics import YOLO

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("smart-scene-analyzer-detection")

    with mlflow.start_run(run_name=a.name):
        mlflow.log_params(params)
        mlflow.set_tag("git_sha", sha)
        mlflow.set_tag("git_dirty", str(dirty).lower())
        if dirty:
            # The SHA alone would describe code that is not what ran.
            mlflow.set_tag(
                "provenance_warning",
                "working tree was dirty; the commit does not fully describe this run",
            )
        mlflow.log_artifact(str(a.data))

        model = YOLO(a.model)

        def log_epoch(trainer: object) -> None:
            """Per-epoch metrics, so a flat or NaN loss is visible while it matters."""
            epoch = int(getattr(trainer, "epoch", 0))
            for key, value in (getattr(trainer, "metrics", None) or {}).items():
                try:
                    clean = key.replace("(", "_").replace(")", "").replace("/", "_")
                    mlflow.log_metric(clean, float(value), step=epoch)
                except (TypeError, ValueError):
                    pass
            for key, value in (getattr(trainer, "label_loss_items", lambda *_: {})(
                getattr(trainer, "tloss", None)
            ) or {}).items():
                with contextlib.suppress(TypeError, ValueError):
                    mlflow.log_metric(key.replace("/", "_"), float(value), step=epoch)

        model.add_callback("on_fit_epoch_end", log_epoch)

        t0 = time.time()
        results = model.train(
            data=str(a.data.resolve()),
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            device=device,
            workers=a.workers,
            name=a.name,
            project=a.project,
            exist_ok=True,
            plots=True,
        )
        elapsed = time.time() - t0

        final = {"train_wall_seconds": elapsed, "seconds_per_epoch": elapsed / max(a.epochs, 1)}
        for key, value in (getattr(results, "results_dict", None) or {}).items():
            try:
                clean = key.replace("(", "_").replace(")", "").replace("/", "_")
                final[f"final_{clean}"] = float(value)
            except (TypeError, ValueError):
                pass
        mlflow.log_metrics(final)

        # Ask ultralytics where it actually wrote, rather than reconstructing the path.
        # It prepends its own settings prefix to `project`, so `--project runs/` lands at
        # `runs/detect/runs/<name>` — reconstructing gives `runs/<name>`, which does not
        # exist, and every artifact is then skipped in silence by the is_file() guard.
        out_dir = Path(
            getattr(results, "save_dir", None)
            or getattr(getattr(model, "trainer", None), "save_dir", None)
            or (Path(a.project) / a.name)
        )
        mlflow.log_param("save_dir", str(out_dir))

        logged = 0
        for artifact in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                         "results.csv", "args.yaml"):
            f = out_dir / artifact
            if f.is_file():
                mlflow.log_artifact(str(f))
                logged += 1
        if logged == 0:
            print(f"WARN  no run artifacts found under {out_dir}", file=sys.stderr)

        print(f"\n{'=' * 68}\nFINAL METRICS\n{'=' * 68}")
        for k, v in final.items():
            print(f"  {k:36s} {v:.5f}")
        print(f"\nweights: {out_dir / 'weights' / 'best.pt'}")
        print(f"mlflow:  run '{a.name}' in experiment 'smart-scene-analyzer-detection'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
