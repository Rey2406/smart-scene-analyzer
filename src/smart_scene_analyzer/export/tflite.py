"""YOLO11 to int8 `.tflite`.

**Single responsibility:** produce the detection artifact the app bundles, and emit the
half of `model_contract.json` that describes it.

int8 quantization needs a representative calibration set drawn from the training
distribution. Calibrating on the wrong images does not fail — it produces an artifact that
loads, runs, and is quietly less accurate than the reference, which is exactly the failure
mode N20 exists to catch. Ultralytics' own TFLite export calibrates from whatever
`data=<data.yaml>` points at (its `val` split, by its own convention) — this module does
not choose a different split; the caller states which one was used, per step 4's explicit
instruction that calibrating on `val` is a real decision, not a default to accept silently.

Nothing here **assumes** the output box encoding or whether coordinates are normalized.
Both are read back from the exported file by actually decoding one real image and checking
which interpretation produces geometrically valid boxes — printed with the evidence, not
asserted from memory, because this export path has changed across ultralytics releases.

**Integration-marked.** It writes files and needs a checkpoint.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_scene_analyzer.export.contract import (
    DetectorInputSpec,
    DetectorManifest,
    DetectorOutputSpec,
    QuantParams,
    sha256_of,
)

LETTERBOX_PAD_VALUE = (114, 114, 114)
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 100


@dataclass
class EncodingProbe:
    """What `_probe_output_encoding` found, and the evidence for it — printed, not hidden."""

    encoding: str  # "cxcywh" | "xyxy"
    normalized: bool
    cxcywh_valid_rate: float
    xyxy_valid_rate: float
    coord_p95: float


def _load_interpreter(tflite_path: Path) -> Any:
    """Prefer the lightweight `ai_edge_litert` runtime; fall back to full TensorFlow."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return interpreter


def _letterbox_to_tensor(image_path: Path, det_size: int, layout: str) -> np.ndarray:
    """Aspect-preserving letterbox, shaped to whichever layout the exported file actually
    reports — **not** assumed. The old TFLite/onnx2tf path conventionally produced NHWC;
    the newer `litert_torch` direct-from-PyTorch path was found (by running this) to
    preserve PyTorch's native NCHW instead. `layout` is read from the real
    `input_details["shape"]` by the caller, so this function never guesses.

    Matches A4/A2: pad value (114,114,114), padding split evenly on both sides. A local,
    minimal reimplementation — `geometry.letterbox` is still a stub (planned public
    surface only) as of this export, so this function does not depend on it.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(det_size / w, det_size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (det_size, det_size), LETTERBOX_PAD_VALUE)
    pad_x, pad_y = (det_size - new_w) // 2, (det_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.uint8)  # HWC
    if layout == "NCHW":
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[None, ...]


def _probe_output_encoding(
    interpreter: Any,
    input_details: dict[str, Any],
    output_details: dict[str, Any],
    calibration_image: Path,
    det_size: int,
) -> EncodingProbe:
    """Decode one real image both ways and keep whichever produces valid boxes.

    A detector's raw output is `[1, 4 + num_classes, num_boxes]` or its transpose. The
    4 box columns are either `cxcywh` (center x/y, width, height — width and height are
    always >= 0 by construction) or `xyxy` (x1,y1,x2,y2, which requires x2>=x1, y2>=y1).
    Reinterpreting one as the other produces negative-area boxes on a large fraction of
    real detections, so the valid-box rate under each interpretation is real evidence,
    not a coin flip.
    """
    in_shape = list(input_details["shape"])
    layout = "NCHW" if len(in_shape) == 4 and in_shape[1] == 3 else "NHWC"
    tensor = _letterbox_to_tensor(calibration_image, det_size, layout)
    scale, zero_point = input_details.get("quantization", (0.0, 0))
    if scale:
        tensor = np.round(tensor.astype(np.float32) / scale + zero_point).astype(
            input_details["dtype"]
        )
    interpreter.set_tensor(input_details["index"], tensor.astype(input_details["dtype"]))
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details["index"]).astype(np.float32)

    out_scale, out_zero = output_details.get("quantization", (0.0, 0))
    if out_scale:
        raw = (raw.astype(np.float32) - out_zero) * out_scale

    arr = raw[0]  # drop batch
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # -> [num_boxes, channels]
    boxes = arr[:, :4]
    # Only look at the boxes a real confidence filter would keep, so the probe reflects
    # actual detections rather than the thousands of near-zero-confidence anchors.
    conf = arr[:, 4:].max(axis=1) if arr.shape[1] > 4 else np.ones(len(arr))
    keep = conf > DEFAULT_CONF_THRESHOLD
    sample = boxes[keep] if keep.any() else boxes[np.argsort(conf)[-50:]]

    cxcywh_valid = float(np.mean((sample[:, 2] >= 0) & (sample[:, 3] >= 0)))
    xyxy_valid = float(np.mean((sample[:, 2] >= sample[:, 0]) & (sample[:, 3] >= sample[:, 1])))

    encoding = "cxcywh" if cxcywh_valid >= xyxy_valid else "xyxy"
    coord_p95 = float(np.percentile(np.abs(sample), 95))
    normalized = (
        coord_p95 <= 2.0
    )  # pixel space reaches det_size (e.g. 640); normalized stays ~[0,1]

    return EncodingProbe(
        encoding=encoding,
        normalized=normalized,
        cxcywh_valid_rate=cxcywh_valid,
        xyxy_valid_rate=xyxy_valid,
        coord_p95=coord_p95,
    )


def _inline_offset_buffers(tflite_path: Path) -> None:
    """Resolves `litert_torch`/`ai_edge_quantizer`'s out-of-line `Buffer.Offset()`/`Size()`
    constant-tensor convention back into the FlatBuffer's classic inline `data` vector,
    in place.

    `export_litert`'s quantization step (`ai_edge_quantizer.quantizer.Quantizer`) writes
    large weight/bias buffers out-of-line, appended after the FlatBuffer root, so they can
    be mmap'd zero-copy. That's a real, valid TFLite feature — Python's `ai_edge_litert`
    runtime resolves it correctly — but `react-native-fast-tflite`'s pinned
    `TensorFlowLiteC 2.17.0` does not, through either of its model-loading entry points
    (confirmed on-device both ways). `Invoke()` fails with "Input tensor N lacks data" the
    moment it reaches the first such tensor, even though `AllocateTensors()` succeeds (shape
    and dtype are metadata, always present) — see
    docs/parity-evidence/round2-invoke-root-cause.md in the course repo this package comes
    from for the full on-device diagnosis.

    `ai_edge_litert.tools.flatbuffer_utils.read_model` already resolves every
    offset-based buffer into `buffer.data` as a normal part of reading a model (it has to,
    to let callers inspect tensor contents at all); `write_model` only supports writing a
    model generated with the offset convention turned off. Round-tripping through this
    pair is therefore a real fix, not a workaround -- confirmed byte-for-byte identical
    inference output (same all-zero input, `ai_edge_litert.Interpreter`, before vs. after)
    and confirmed on-device via `Invoke()` succeeding where it previously failed, every
    time, consistent with the Python reference within ordinary int8/XNNPACK
    cross-platform noise (~0.08% relative, mean output value).
    """
    from ai_edge_litert.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(tflite_path)
    flatbuffer_utils.write_model(model, tflite_path)


def export_detector(
    checkpoint: Path,
    out_path: Path,
    data_yaml: Path,
    det_size: int = 640,
    registry_version: str = "unversioned",
    calibration_split: str = "val",
) -> tuple[DetectorManifest, EncodingProbe]:
    """Export `checkpoint` to int8 TFLite at `out_path`, reading shapes back from the file.

    Returns the manifest fragment AND the `EncodingProbe` evidence, so the caller can print
    exactly how the box encoding was determined rather than take it on faith.
    """
    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(str(checkpoint))
    # format='tflite'/int8=True are deprecated as of ultralytics 8.4.118 in favor of
    # format='litert' (Google's TFLite rebrand) / quantize=8 (static int8, requires the
    # calibration `data` below) — found by running the export, not from documentation.
    exported = model.export(format="litert", quantize=8, data=str(data_yaml), imgsz=det_size)
    exported_path = Path(exported)
    if exported_path.is_dir():
        candidates = sorted(exported_path.glob("*_integer_quant.tflite"))
        if not candidates:
            candidates = sorted(exported_path.glob("*.tflite"))
        if not candidates:
            raise FileNotFoundError(f"No .tflite file found under {exported_path}")
        exported_path = candidates[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported_path, out_path)
    _inline_offset_buffers(out_path)

    interpreter = _load_interpreter(out_path)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    quant = None
    scale, zero_point = input_details.get("quantization", (0.0, 0))
    if scale:
        quant = QuantParams(scale=float(scale), zero_point=int(zero_point))

    with data_yaml.open() as f:
        data_cfg = yaml.safe_load(f)
    labels = [data_cfg["names"][i] for i in sorted(data_cfg["names"])]

    calibration_images = (
        sorted(
            (data_yaml.parent / calibration_split.replace("val", "valid") / "images").glob("*.jpg")
        )[:1]
        or sorted((data_yaml.parent / "test" / "images").glob("*.jpg"))[:1]
    )
    if not calibration_images:
        raise FileNotFoundError("No image found to probe the exported output encoding.")
    probe = _probe_output_encoding(
        interpreter, input_details, output_details, calibration_images[0], det_size
    )

    manifest = DetectorManifest(
        file=out_path.name,
        sha256=sha256_of(out_path),
        registry_version=registry_version,
        input=DetectorInputSpec(
            shape=list(input_details["shape"]),
            dtype=str(input_details["dtype"]),
            layout=(
                "NCHW"
                if len(input_details["shape"]) == 4 and input_details["shape"][1] == 3
                else "NHWC"
            ),
            quant=quant,
            normalization="none (raw uint8 pixels, quantized per `quant`)"
            if quant
            else "[0,1] float",
            letterbox_pad_value=list(LETTERBOX_PAD_VALUE),
        ),
        outputs=[
            DetectorOutputSpec(
                index=0,
                name=str(output_details.get("name", "output_0")),
                shape=list(output_details["shape"]),
                dtype=str(output_details["dtype"]),
                encoding=probe.encoding,  # type: ignore[arg-type]
                normalized=probe.normalized,
            )
        ],
        labels=labels,
        default_conf_threshold=DEFAULT_CONF_THRESHOLD,
        default_iou_threshold=DEFAULT_IOU_THRESHOLD,
        max_detections=DEFAULT_MAX_DETECTIONS,
    )
    return manifest, probe
