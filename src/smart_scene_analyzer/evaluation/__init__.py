"""Metrics: N3 detection accuracy and N4 depth ordering.

**Single responsibility:** measure, and report. Nothing in this package changes anything it
measures.

That boundary is deliberate and it is mirrored in the harness: the Evaluation role has no
`Edit` tool. An agent — or a module — that can adjust the model when it dislikes the number
is not measuring anything. So this package reads checkpoints, artifacts, and predictions,
and writes reports.

Two metrics, and the second one is a deliberate refusal:

- **N3, mAP@50 >= 0.50 (provisional).** Provisional because the object taxonomy is an
  **open decision** (Lesson 02), and an mAP target over an unknown number of classes is
  not a target. Nothing here assumes a class count.
- **N4, pairwise depth-ordering accuracy >= 0.90.** Not the usual threshold metric, which
  compares predictions to ground truth on a metric scale this system does not have.
  Applying it would require fitting a scale and shift per image, which invents the very
  quantity the project refuses to claim. What the product relies on is **ordering** — that
  a nearer object scores as nearer — so that is what is measured. A metric already assumes
  an answer to the question the architecture declined to ask.

**Integration-marked** wherever a checkpoint or an artifact is needed. The scoring
functions themselves are pure and always tested: predictions and labels in, numbers out.
"""
