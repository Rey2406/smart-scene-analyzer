"""Model-card inputs: metrics assembled into something a person can be held to.

**Single responsibility:** turn metric objects into a report that names its conditions.
It formats; it does not compute a metric and it does not change a model.

A number without its conditions is not comparable to the next number, so every report
states: the dataset version, the model registry version, which artifact produced it (the
fp32 reference or the quantized export — they are different models and reporting one as
the other is the most common way this project could mislead itself), the device or host it
ran on, and the split.

The report also carries what the system **cannot** do, because a model card that omits its
limits is marketing:

- no absolute depth scale, so no distance may be quoted anywhere;
- depth values comparable only within one image;
- the fusion reduction's known failure on boxes that are mostly background, with the
  `depth_spread` stratification from `depth_ordering` as the evidence;
- N3's target is provisional while the taxonomy is an open decision.

Planned public surface:

    def build(
        detection: DetectionMetrics,
        ordering: OrderingMetrics,
        conditions: ReportConditions,
    ) -> ModelCard

    def render_markdown(card: ModelCard) -> str
"""
