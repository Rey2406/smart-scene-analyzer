"""YOLO11 to int8 `.tflite`.

**Single responsibility:** produce the detection artifact the app bundles, and emit the
half of `model_contract.json` that describes it.

int8 quantization needs a representative calibration set drawn from the training
distribution. Calibrating on the wrong images does not fail — it produces an artifact that
loads, runs, and is quietly less accurate than the reference, which is exactly the failure
mode N20 exists to catch.

Record the fp32 size alongside the int8 size in `docs/artifact-budget.md`. The delta is
the quantization's entire justification, and a size recorded without its counterfactual
does not show whether it was worth the accuracy risk. N9a caps the two artifacts combined
at 40 MB.

**Integration-marked.** It writes files and needs a checkpoint.

Whether this exporter is the one that runs at all is an **open decision**: if two native
runtimes prove unaffordable, one consolidation branch replaces this with a `.pte` export
of YOLO11 (`docs/artifact-budget.md`, architecture §10.4). Nothing in this module's
interface presumes the answer.

Planned public surface:

    def export_detector(
        checkpoint: Path,
        out_path: Path,
        calibration_images: Sequence[Path],
        det_size: int,
    ) -> DetectorArtifactSpec
        \"\"\"-> the manifest fragment for this artifact: input shape/dtype/layout, quant
        parameters, output specs, label order, and sha256.\"\"\"
"""
