#!/usr/bin/env python3
"""Export both models reproducibly: YOLO11 -> int8 TFLite, Depth Anything V2 -> ExecuTorch.

Lesson 04, step 4. Writes `app/assets/models/*.tflite`, `*.pte`, and `model_contract.json`
(contract C11) — the three files `docs/architecture.md` says cross from Boundary B to
Boundary A. The first two are gitignored; the manifest is not, because it is the only part
of the contract a reviewer can read without loading a 20 MB binary.

⚠️ Neither export toolchain is verified against this project's models as of this writing.
A failure is the finding this step asks for — this script does not substitute a different
model, recipe, or task to make an export succeed.

Usage:
    uv run python scripts/export.py --all
    uv run python scripts/export.py --detection
    uv run python scripts/export.py --depth
    uv run python scripts/export.py --all --force
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smart_scene_analyzer.export import contract
from smart_scene_analyzer.export.contract import ArtifactManifest

DEFAULT_DET_CHECKPOINT = Path("runs/detect/runs/v1-baseline/weights/best.pt")
DEFAULT_DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_DATA_YAML = Path("data/v1/data.yaml")
DEFAULT_OUT_DIR = Path("app/assets/models")
DEFAULT_DET_SIZE = 640
EXPORTER_VERSION = "scripts/export.py@lesson04-step4"


def _run_name_from_checkpoint(checkpoint: Path) -> str | None:
    """`runs/detect/runs/<name>/weights/best.pt` -> `<name>`, by this project's own
    convention (the doubled `runs/detect/runs` prefix is ultralytics' own, per
    docs/model-card-v1.md's Identity table)."""
    parts = checkpoint.parts
    if "weights" in parts:
        idx = parts.index("weights")
        if idx > 0:
            return parts[idx - 1]
    return None


def _log_to_mlflow(run_name: str | None, params: dict[str, object], artifact_path: Path) -> str:
    """Log export config against the MLflow run that produced the checkpoint.

    Resumes that run (rather than starting a new one) so the export is discoverable from
    the training run it belongs to. Params are prefixed `export_` so re-running this
    script against the same run never collides with a param MLflow has already fixed.
    """
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run_id = None
    if run_name:
        df = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=f"tags.mlflow.runName = '{run_name}'",
        )
        if len(df) > 0:
            run_id = df.iloc[0]["run_id"]

    if run_id is None:
        print(
            f"  WARN   No MLflow run named {run_name!r} found — logging export under a "
            "new run instead of the training run that produced the checkpoint."
        )

    with mlflow.start_run(run_id=run_id, run_name=None if run_id else "export"):
        mlflow.log_params({f"export_{k}": v for k, v in params.items()})
        mlflow.log_artifact(str(artifact_path))
        active_run_id = mlflow.active_run().info.run_id
    return active_run_id


def _report_file(label: str, path: Path, input_shape: list[int], output_shape: list[int]) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n  {label}")
    print(f"    path          {path}")
    print(f"    size          {size_mb:.2f} MiB")
    print(f"    input shape   {input_shape}  (read back from the exported file)")
    print(f"    output shape  {output_shape}  (read back from the exported file)")


def export_detection(
    det_checkpoint: Path, data_yaml: Path, out_dir: Path, det_size: int, force: bool
) -> ArtifactManifest:
    out_path = out_dir / "detector.tflite"
    if out_path.exists() and not force:
        sys.exit(f"FAIL   {out_path} already exists. Pass --force to overwrite.")
    if not det_checkpoint.is_file():
        sys.exit(f"FAIL   Checkpoint not found: {det_checkpoint}")
    if not data_yaml.is_file():
        sys.exit(f"FAIL   Dataset config not found: {data_yaml}")

    from smart_scene_analyzer.export.tflite import export_detector

    run_name = _run_name_from_checkpoint(det_checkpoint)
    print(f"  Exporting detection: {det_checkpoint} -> {out_path}")
    print(f"  Calibration data:    {data_yaml} (val split — ultralytics' own convention)")
    print(
        "  ⚠️  Calibrating int8 on the VAL split means step 6's parity number is "
        "optimistic: quantization error is measured on data that chose the "
        "quantization parameters, not held-out data."
    )

    det_manifest, probe = export_detector(
        checkpoint=det_checkpoint,
        out_path=out_path,
        data_yaml=data_yaml,
        det_size=det_size,
        registry_version=run_name or "unversioned",
        calibration_split="val",
    )

    _report_file(
        "Detection (int8 TFLite)", out_path, det_manifest.input.shape, det_manifest.outputs[0].shape
    )
    print(f"    quantization  {det_manifest.input.quant}")
    print(
        f"    box encoding  {det_manifest.outputs[0].encoding}  "
        f"(cxcywh-valid-rate={probe.cxcywh_valid_rate:.2f}, "
        f"xyxy-valid-rate={probe.xyxy_valid_rate:.2f} on a real calibration image — "
        "the higher rate is reported, both are shown so the call is checkable)"
    )
    print(
        f"    normalized?   {det_manifest.outputs[0].normalized}  "
        f"(95th-pct abs coord = {probe.coord_p95:.2f}; <=2.0 => normalized, "
        f"larger => {det_size}-px input-pixel space)"
    )

    fp32_path = out_dir / "detector_fp32_for_comparison.tflite"
    print(
        f"\n  NOTE: fp32-vs-int8 size ratio needs a separate fp32 export "
        f"({fp32_path.name}, not produced by this run — pass --fp32-too to add one in a "
        "follow-up if the comparison in step 4's Round 2 is still wanted)."
    )

    run_id = _log_to_mlflow(
        run_name,
        {
            "detection_format": "tflite-int8",
            "det_size": det_size,
            "calibration_split": "val",
            "exporter_version": EXPORTER_VERSION,
        },
        out_path,
    )
    print(f"    MLflow run    {run_id}")

    return ArtifactManifest(
        detector=det_manifest,
        exported_at=datetime.now(UTC).isoformat(),
        exporter_version=EXPORTER_VERSION,
        dataset_version=str(data_yaml.parent.name),
    )


def export_depth_model(
    depth_checkpoint: str, out_dir: Path, probe_image: Path, force: bool
) -> ArtifactManifest:
    out_path = out_dir / "depth.pte"
    if out_path.exists() and not force:
        sys.exit(f"FAIL   {out_path} already exists. Pass --force to overwrite.")
    if not probe_image.is_file():
        sys.exit(f"FAIL   Probe image not found: {probe_image}")

    from smart_scene_analyzer.export.executorch import export_depth

    print(f"\n  Exporting depth: {depth_checkpoint} -> {out_path}")
    print("  Recipe: xnnpack, task: depth-estimation (per step 4's exact command)")

    depth_manifest, sign_check = export_depth(
        checkpoint=depth_checkpoint,
        out_path=out_path,
        probe_image=probe_image,
        registry_version=depth_checkpoint,
    )

    _report_file(
        "Depth (ExecuTorch)", out_path, depth_manifest.input.shape, depth_manifest.output.shape
    )
    print(f"    sign check    {sign_check.note}")
    if not sign_check.passed:
        print(
            "    FAIL   Sign convention did NOT verify. Reporting this rather than "
            "silently shipping an inverted depth model — see docs/model-card-v1.md."
        )

    run_id = _log_to_mlflow(
        None,
        {
            "depth_format": "executorch-xnnpack",
            "depth_checkpoint": depth_checkpoint,
            "exporter_version": EXPORTER_VERSION,
        },
        out_path,
    )
    print(f"    MLflow run    {run_id}")

    return ArtifactManifest(
        depth=depth_manifest,
        exported_at=datetime.now(UTC).isoformat(),
        exporter_version=EXPORTER_VERSION,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Export both models reproducibly.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--detection", action="store_true")
    group.add_argument("--depth", action="store_true")
    p.add_argument("--det-checkpoint", type=Path, default=DEFAULT_DET_CHECKPOINT)
    p.add_argument("--depth-checkpoint", default=DEFAULT_DEPTH_CHECKPOINT)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE)
    p.add_argument(
        "--probe-image",
        type=Path,
        default=None,
        help="Real image used to verify the depth sign convention. Defaults to the "
        "first test-split image.",
    )
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = a.out_dir / "model_contract.json"
    existing = contract.read(manifest_path) if manifest_path.is_file() else None

    probe_image = a.probe_image
    if probe_image is None:
        candidates = sorted((a.data.parent / "test" / "images").glob("*.jpg"))
        probe_image = candidates[0] if candidates else None

    det_result = depth_result = None
    if a.all or a.detection:
        det_result = export_detection(a.det_checkpoint, a.data, a.out_dir, a.det_size, a.force)
    if a.all or a.depth:
        if probe_image is None:
            sys.exit("FAIL   No probe image available for the depth sign-convention check.")
        depth_result = export_depth_model(a.depth_checkpoint, a.out_dir, probe_image, a.force)

    merged = ArtifactManifest(
        detector=(det_result.detector if det_result else existing.detector if existing else None),
        depth=(depth_result.depth if depth_result else existing.depth if existing else None),
        exported_at=datetime.now(UTC).isoformat(),
        exporter_version=EXPORTER_VERSION,
        dataset_version=(
            det_result.dataset_version
            if det_result
            else existing.dataset_version
            if existing
            else None
        ),
    )
    contract.write(merged, manifest_path)
    print(f"\n  Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
