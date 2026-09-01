"""The YOLO11 fine-tuning entry point.

**Single responsibility:** run one reproducible training run and hand back a checkpoint
plus the record of how it was made.

Reproducibility is the deliverable alongside the weights (N19): the seed, the dataset
version, and the full resolved config are recorded through the `RunTracker` before the run
starts, not summarized after it finishes. A run whose config was reconstructed from memory
afterwards is not reproducible, whatever its metrics say.

The training target is shaped by what the artifact has to become. A model that trains well
and quantizes badly is not a usable outcome here — the app bundles int8, so accuracy is
whatever survives `export.tflite`, and N9a caps the two artifacts at 40 MB combined. That
argues for a small YOLO11 variant chosen up front rather than a large one shrunk later.

**Integration-marked.** GPU, dataset, time, and possibly Roboflow credits — which require
an estimate and approval first (`docs/credit-budget.md`).

Planned public surface:

    def train(
        splits: DatasetSplits,
        config: TrainingConfig,     # includes the seed
        tracker: RunTracker,
    ) -> TrainingResult
        \"\"\"-> checkpoint path, final metrics, and the tracker's run id.\"\"\"
"""
