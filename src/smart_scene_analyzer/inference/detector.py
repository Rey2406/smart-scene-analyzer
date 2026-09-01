"""The reference YOLO11 detection backend (`DetectorBackend`).

**Single responsibility:** be the PyTorch detector that the exported int8 `.tflite`
artifact is proved equivalent to (N20), and nothing else.

This is a reference, not a product. It never runs on a handset. Its only jobs are to
produce the numbers the export is compared against and to give evaluation something
unquantized to attribute a metric drop to.

**Impure.** It loads weights and imports `ultralytics`. Every test that touches it carries
`@pytest.mark.integration`, and the default suite does not run it. Code that wants a
detector in a unit test depends on `DetectorBackend` from `protocols.py` and passes a
fixture-backed double.

Planned public surface:

    class ReferenceDetector:
        \"\"\"Implements DetectorBackend over an Ultralytics YOLO11 checkpoint.\"\"\"
        def __init__(self, checkpoint: Path, registry_version: str) -> None: ...
        @property
        def registry_version(self) -> str: ...
        def infer(self, model_input: DetectionInput) -> RawDetectorOutput: ...

The class list this model predicts is an **open decision** (Lesson 02). Nothing in this
module encodes one; labels are resolved from the artifact manifest by `decode.py`.
"""
