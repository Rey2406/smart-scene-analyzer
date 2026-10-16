"""Depth Anything V2 to ExecuTorch `.pte`.

**Single responsibility:** produce the depth artifact the app bundles, and emit the half of
`model_contract.json` that describes it.

The vision transformer is the harder of the two artifacts to shrink, and it is the one that
dominates both N9a (40 MB combined) and N6 (the 1.4 s of the 3.0 s cold-load budget
allocated to deserializing it).

The exported graph must preserve the output's meaning as well as its shape: relative
inverse depth, larger nearer, no unit and no scale. An export that flips polarity or
rescales produces a map that passes a shape check and renders every scene inside out.
That is why the manifest carries an explicit `semantics` string the app asserts on at load
(`export.contract`). This module verifies the convention against the PyTorch reference on
a real image rather than assuming the export tool preserved it.

**Integration-marked.** It writes files and needs a checkpoint.

⚠️ Neither this toolchain nor YOLO11's is verified against this project's models as of
this writing. If the export fails or the artifact shape is not the fixed `[1,3,518,518]`
input / native `[518,518]` output this project's contract requires, that failure IS the
finding — this module does not substitute a different checkpoint, recipe, or task to make
it pass.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smart_scene_analyzer.export.contract import (
    DepthInputSpec,
    DepthManifest,
    DepthOutputSpec,
    sha256_of,
)

DEPTH_SIZE = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SignConventionCheck:
    """Evidence that larger-is-nearer survived the export, not an assumption of it."""

    passed: bool
    reference_rank_correlation: float  # Spearman rho between exported and PyTorch reference
    note: str


def _run_optimum_cli_export(checkpoint: str, output_dir: Path) -> None:
    cmd = [
        "optimum-cli",
        "export",
        "executorch",
        "--model",
        checkpoint,
        "--task",
        "depth-estimation",
        "--recipe",
        "xnnpack",
        "--output_dir",
        str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"optimum-cli export executorch failed (exit {result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def _load_pte_metadata(pte_path: Path) -> tuple[list[int], list[int], str, str]:
    """Input shape, output shape, input dtype, output dtype — read from the .pte itself."""
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(str(pte_path))
    method = program.load_method("forward")
    meta = method.metadata
    in_spec = meta.input_tensor_meta(0)
    out_spec = meta.output_tensor_meta(0)
    return (
        list(in_spec.sizes()),
        list(out_spec.sizes()),
        str(in_spec.dtype()),
        str(out_spec.dtype()),
    )


def _run_pte(pte_path: Path, tensor: np.ndarray) -> np.ndarray:
    import torch
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(str(pte_path))
    method = program.load_method("forward")
    (out,) = method.execute([torch.from_numpy(tensor)])
    result: np.ndarray = out.numpy()
    return result


def _letterbox_square(image_path: Path, size: int) -> np.ndarray:
    """`[1, 3, size, size]` float32 NCHW, ImageNet-normalized (A3). No external dep on the
    not-yet-implemented `geometry`/`preprocess` stub modules."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None, ...]  # HWC -> CHW, add batch


def _verify_sign_convention(pte_path: Path, probe_image: Path) -> SignConventionCheck:
    """Run the SAME image through the exported .pte and the PyTorch reference, and check
    the two rank-correlate positively. A convention flip shows up as strong NEGATIVE
    correlation — still structured, still plausible-looking, and exactly backwards."""
    from smart_scene_analyzer.depth import estimate_relative_inverse_depth

    reference = estimate_relative_inverse_depth(Path(probe_image))
    tensor = _letterbox_square(probe_image, DEPTH_SIZE)
    exported_raw = _run_pte(pte_path, tensor)

    exported = np.asarray(exported_raw).squeeze()
    ref_resized = (
        np.array(
            __import__("PIL.Image", fromlist=["Image"])
            .fromarray(reference)
            .resize((exported.shape[-1], exported.shape[-2]))
        )
        if reference.shape != exported.shape
        else reference
    )

    rho = float(
        np.corrcoef(exported.ravel().argsort().argsort(), ref_resized.ravel().argsort().argsort())[
            0, 1
        ]
    )
    passed = rho > 0.5
    note = (
        f"Spearman rho between exported .pte output and the PyTorch reference, both "
        f"treated as relative inverse depth (larger=nearer), on {probe_image.name}: "
        f"{rho:.3f}. {'Positive' if rho > 0 else 'NEGATIVE'} correlation "
        f"{'confirms' if passed else 'CONTRADICTS'} the sign convention survived export."
    )
    return SignConventionCheck(passed=passed, reference_rank_correlation=rho, note=note)


def export_depth(
    checkpoint: str,
    out_path: Path,
    probe_image: Path,
    registry_version: str = "unversioned",
) -> tuple[DepthManifest, SignConventionCheck]:
    """Export `checkpoint` to ExecuTorch at `out_path`. Raises with the tool's actual
    stdout/stderr on failure — never substitutes a different checkpoint/recipe/task."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _run_optimum_cli_export(checkpoint, tmp_dir)
        candidates = sorted(tmp_dir.rglob("*.pte"))
        if not candidates:
            raise FileNotFoundError(
                f"optimum-cli reported success but no .pte found under {tmp_dir}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(candidates[0], out_path)

    in_shape, out_shape, in_dtype, out_dtype = _load_pte_metadata(out_path)

    if list(in_shape) != [1, 3, DEPTH_SIZE, DEPTH_SIZE]:
        raise ValueError(
            f"Exported .pte input shape is {in_shape}, not the fixed "
            f"[1, 3, {DEPTH_SIZE}, {DEPTH_SIZE}] this project's contract (C4) requires. "
            "This is the finding, not a bug to work around — report it rather than "
            "resizing around it."
        )

    sign_check = _verify_sign_convention(out_path, probe_image)

    manifest = DepthManifest(
        file=out_path.name,
        sha256=sha256_of(out_path),
        registry_version=registry_version,
        input=DepthInputSpec(
            shape=list(in_shape),
            dtype=in_dtype,
            layout="NCHW",
            normalization=f"ImageNet mean={IMAGENET_MEAN} std={IMAGENET_STD}",
        ),
        output=DepthOutputSpec(
            shape=list(out_shape),
            dtype=out_dtype,
            semantics="relative-inverse-depth-larger-is-nearer",
        ),
    )
    return manifest, sign_check
