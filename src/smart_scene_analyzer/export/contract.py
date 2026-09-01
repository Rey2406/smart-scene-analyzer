"""`model_contract.json` — write it, read it, validate it (contract C11).

**Single responsibility:** be the single written statement of everything the app assumes
about the artifacts, so that those assumptions are reviewable instead of implicit.

The export **defines** this; the app **consumes** it. Every field below is fixed on this
side of the boundary and taken on faith on the other:

- input tensor shape, dtype, and layout, per model — `[1, 640, 640, 3]` NHWC is not
  `[1, 3, 640, 640]` NCHW, and a swap produces plausible garbage rather than an error;
- the quantization scale and zero point for the int8 detection input;
- channel normalization constants;
- the letterbox pad value;
- output tensor count, order, shape, dtype, box encoding (`cxcywh` or `xyxy`), and whether
  coordinates are normalized;
- **the ordered label list.** This is the highest-risk field in the project. An app built
  against a stale order runs, draws correct boxes, and names every one of them wrong, and
  no test on either side catches it. `sha256` plus `artifact_drift.py` is the defence;
- the default confidence and IoU thresholds and the per-image detection cap, so that the
  app and the reference use one set of numbers rather than two constants in two languages
  (assumptions A7, A8);
- the model registry version for each model, which is the only on-device source for F7 —
  there is no service to ask;
- `depth.output.semantics`, a literal string asserting that the depth output is relative
  inverse depth with larger meaning nearer. The app checks it at load, so that an export
  which ever flips polarity fails loudly instead of rendering every scene inside out.

The label list's contents are an **open decision** (the taxonomy, Lesson 02). This module
records whatever order it is given and computes nothing from its length.

Planned public surface:

    class ArtifactManifest(BaseModel)    # the schema of C11, validated on read
    def write(manifest: ArtifactManifest, path: Path) -> None
    def read(path: Path) -> ArtifactManifest
    def verify_files(manifest: ArtifactManifest, models_dir: Path) -> list[str]
        \"\"\"-> human-readable mismatches: a missing file, or a sha256 that does not match
        the manifest. Empty list means the bundled artifacts are the ones described.\"\"\"
"""
