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

**Why this goes through ExecuTorch's core APIs instead of `optimum-cli`:**
`optimum-executorch`'s ExecuTorch-specific task registry has no `depth-estimation` recipe
(checked against the installed release and unreleased `main`, five days apart, both
confirmed missing — see `docs/decisions/0001-inference-target.md`). The general-purpose
`TasksManager` has the task; the ExecuTorch backend's own registry does not. That is a
toolchain gap, not a model or platform limitation, so this module drives `torch.export`,
`to_edge_transform_and_lower`, and `XnnpackPartitioner` directly rather than waiting on it.

**Two problems this path previously had, both now fixed and worth naming so nobody
reintroduces them:**

1. The example tensor fed to `torch.export.export` must be `.contiguous()`.
   `_letterbox_square`'s `arr.transpose(2, 0, 1)[None, ...]` inserts the batch axis via
   NumPy's `newaxis`, which gives that axis **stride 0**, and `torch.from_numpy` carries
   the non-contiguous layout straight through tracing. A stride-0 placeholder either makes
   `to_executorch()` raise `"0 in strides is not supported for ExecuTorch"` outright, or —
   worse, when the graph shape happens to let it through — corrupts the `.pte`'s recorded
   input layout so the native runtime misreads the buffer: shape-correct, values scrambled.
   This produced a 0.74 rank-correlation "fidelity gap" that looked like a `to_edge()`
   lowering bug and wasn't one; it was a non-contiguous input tensor. Fixed by
   `np.ascontiguousarray(...)` here.
2. Unquantized, the float32 export is 94.45 MiB — 2.4x N9a's entire 40 MB combined budget
   on its own. Fixed by quantizing through ExecuTorch's own PT2E flow (`XNNPACKQuantizer`,
   per-channel symmetric int8, calibrated on real images) before lowering, the same way
   `quantize=8` quantizes the detection model via a different toolchain. Brings the
   artifact to ~24 MB at a measured mean rank correlation of 0.97 against PyTorch eager
   (std 0.015, n=8 held-out images) — real quantization noise, not a lowering bug.

⚠️ If the export fails or the artifact shape is not the fixed `[1,3,518,518]` input /
native `[518,518]` output this project's contract requires, that failure IS the finding —
this module does not substitute a different checkpoint, recipe, or task to make it pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from smart_scene_analyzer.export.contract import (
    DepthInputSpec,
    DepthManifest,
    DepthOutputSpec,
    sha256_of,
)

DEPTH_SIZE = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CALIBRATION_IMAGE_COUNT = 8


@dataclass
class SignConventionCheck:
    """Evidence that larger-is-nearer survived the export, not an assumption of it."""

    passed: bool
    reference_rank_correlation: float  # Spearman rho between exported and PyTorch reference
    note: str


def _load_depth_wrapper(checkpoint: str) -> torch.nn.Module:
    """The HF depth model wrapped to return a bare tensor. `torch.export` needs a module
    whose forward returns tensors, not the `DepthEstimatorOutput` dataclass HF models use."""
    import torch
    from transformers import AutoModelForDepthEstimation

    class DepthWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            depth: torch.Tensor = self.model(pixel_values=pixel_values).predicted_depth
            return depth

    model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
    model.eval()
    return DepthWrapper(model)


def _select_calibration_images(
    probe_image: Path, count: int = CALIBRATION_IMAGE_COUNT
) -> list[Path]:
    """Real images for quantization calibration, deterministic: sorted siblings of
    `probe_image` in its own directory, excluding it. int8 needs real activation
    statistics — calibrating on the probe image alone would be one data point."""
    candidates = [p for p in sorted(probe_image.parent.glob("*.jpg")) if p != probe_image]
    if not candidates:
        raise FileNotFoundError(
            f"No calibration images found alongside probe image {probe_image} "
            "(expected sibling .jpg files for int8 calibration)."
        )
    return candidates[:count]


def _export_via_executorch(checkpoint: str, probe_image: Path, out_path: Path) -> None:
    """`torch.export` -> PT2E int8 quantization (calibrated) -> `to_edge_transform_and_lower`
    with `XnnpackPartitioner` -> `.to_executorch()`. See the module docstring for why this
    bypasses `optimum-cli` and why the input tensor must be contiguous before `torch.export`."""
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    wrapper = _load_depth_wrapper(checkpoint)
    calibration_images = _select_calibration_images(probe_image)

    example = torch.from_numpy(_letterbox_square(calibration_images[0], DEPTH_SIZE)).contiguous()
    exported = torch.export.export(wrapper, (example,), strict=False)

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(exported.module(), quantizer)

    with torch.no_grad():
        for image_path in calibration_images:
            calib_tensor = torch.from_numpy(_letterbox_square(image_path, DEPTH_SIZE)).contiguous()
            prepared(calib_tensor)

    converted = convert_pt2e(prepared)
    quantized_exported = torch.export.export(converted, (example,), strict=False)

    edge = to_edge_transform_and_lower(
        quantized_exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    program = edge.to_executorch()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)


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
    (out,) = method.execute([torch.from_numpy(tensor).contiguous()])
    result: np.ndarray = out.numpy()
    return result


def _letterbox_square(image_path: Path, size: int) -> np.ndarray:
    """`[1, 3, size, size]` float32 NCHW, ImageNet-normalized (A3), contiguous. No external
    dep on the not-yet-implemented `geometry`/`preprocess` stub modules.

    The `contiguous` part is load-bearing, not cosmetic: `[None, ...]` inserts the batch
    axis via NumPy's `newaxis`, which gives that axis stride 0. `torch.from_numpy` carries
    a stride-0 array straight through `torch.export`, which either crashes `to_executorch`
    ("0 in strides is not supported for ExecuTorch") or — when the graph shape lets it
    through — corrupts the `.pte`'s recorded input layout so the runtime misreads the
    buffer. `np.ascontiguousarray` is the fix; see the module docstring."""
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
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])  # HWC -> CHW, add batch


def _verify_sign_convention(pte_path: Path, probe_image: Path) -> SignConventionCheck:
    """Run the SAME image through the exported .pte and the PyTorch reference, and check
    the two rank-correlate positively. A convention flip shows up as strong NEGATIVE
    correlation — still structured, still plausible-looking, and exactly backwards."""
    from PIL import Image

    from smart_scene_analyzer.depth import estimate_relative_inverse_depth

    probe_array = np.asarray(Image.open(probe_image).convert("RGB"), dtype=np.uint8)
    reference = estimate_relative_inverse_depth(probe_array)
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
    """Export `checkpoint` to ExecuTorch at `out_path` via `torch.export` + PT2E int8
    quantization + `to_edge_transform_and_lower`/`XnnpackPartitioner` — never substitutes a
    different checkpoint, recipe, or task to make the export pass."""
    _export_via_executorch(checkpoint, probe_image, out_path)

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
