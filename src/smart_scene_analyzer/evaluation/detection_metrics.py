"""N3 — detection accuracy: mAP@50 and the per-class breakdown behind it.

**Single responsibility:** score predicted boxes against ground-truth boxes. Pure: boxes
in, numbers out. No weights, no files, always tested.

Both predictions and ground truth must be `xyxy` in absolute pixels of the **`SOURCE`**
space, and the scorer asserts it rather than assuming it. Scoring predictions in
`DET_INPUT` against labels in `SOURCE` produces a number — a low one, plausibly explained
as a weak model — and that is the worst kind of wrong answer this project can get.

**The aggregate is not the finding; the per-class breakdown is.** A single mAP figure
cannot distinguish a class with too few instances (DATA), a class that overlaps another in
the label definition (TAXONOMY), and a class the model genuinely cannot separate (MODEL) —
which is the classification `.claude/skills/error-triage` exists to make. So the per-class
table is the primary return value and the aggregate is derived from it.

The class count is unknown: the taxonomy is an **open decision**. Nothing here is written
against a fixed number of classes.

Planned public surface:

    def evaluate(
        predictions: Sequence[Sequence[Detection]],   # xyxy absolute px, SOURCE space
        ground_truth: Sequence[Sequence[Detection]],  # xyxy absolute px, SOURCE space
        iou_threshold: float = 0.50,
    ) -> DetectionMetrics
        \"\"\"-> mAP@50, plus per-class precision, recall, AP, and instance counts.\"\"\"

    def confusion(
        predictions: Sequence[Sequence[Detection]],
        ground_truth: Sequence[Sequence[Detection]],
        iou_threshold: float = 0.50,
    ) -> ConfusionMatrix
        \"\"\"Includes explicit background rows and columns: a missed object and a
        false positive are different errors and must not net out.\"\"\"
"""
