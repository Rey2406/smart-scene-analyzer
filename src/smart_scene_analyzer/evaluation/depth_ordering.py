"""N4 — pairwise depth-ordering accuracy over annotated object pairs.

**Single responsibility:** measure whether the system gets the *order* right, which is the
only depth claim this system makes.

The metric: over every annotated pair of objects in an image where the annotation states
which is nearer, count the fraction of pairs whose predicted `relative_depth` agrees.
Target >= 0.90 on the held-out test split.

**Why this and not a threshold-ratio metric.** Those compare a prediction to ground truth
on an absolute scale, and this system has no absolute scale to compare. Using one would
require fitting a per-image scale and shift, which manufactures exactly the quantity the
project refuses to claim, and then reports an error against the manufactured version. The
product's actual requirement is that a nearer object scores as nearer, so that is the
requirement measured. Depth here is relative inverse depth: larger is nearer, unit-less,
comparable only within one image — so a *within-image, pairwise, ordinal* metric is the
only kind that is even well-defined.

**Report ties explicitly.** Two objects at genuinely similar depth are a case the system
cannot be expected to order, and folding them into the pass count inflates the score while
folding them into the fail count deflates it. They are counted and reported separately,
with the ambiguity band stated.

**Stratify by `depth_spread`.** The fusion layer's median reduction fails on boxes that
are mostly background, and it emits `depth_spread` precisely so that failure is visible.
Ordering accuracy split by spread says whether N4's misses are the ones the reduction
already admitted to — which is a finding — or somewhere else, which is a different one.

Pure: fused detections and pair annotations in, numbers out. Always tested.

Planned public surface:

    def evaluate(
        predictions: Sequence[Sequence[FusedDetection]],
        pair_annotations: Sequence[Sequence[OrderedPair]],
        tie_band: float = 0.0,
    ) -> OrderingMetrics
        \"\"\"-> overall pairwise ordering accuracy, the tie count, and the accuracy
        stratified by `depth_spread`. All inputs are unit-less relative inverse depth,
        larger nearer, and are only ever compared within one image.\"\"\"
"""
