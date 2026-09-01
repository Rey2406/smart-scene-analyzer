"""Detection fine-tuning and run tracking.

**Single responsibility:** produce a detection checkpoint from a named dataset version, and
record enough about how it was produced that the run can be repeated (N19).

Only the detector is trained. Depth Anything V2 is used as released — the project has no
depth ground truth (which is why N4 measures ordering rather than a metric error), so there
is nothing to fine-tune it against.

Everything in this package is **integration-marked**: it needs a GPU, a dataset, and time.
Nothing else in `src/` imports it, so the default offline suite never touches it.

Two constraints from outside this package shape it:

- **The dataset version is a version, never "the latest."** A Roboflow dataset version is
  an immutable snapshot, and a run that cannot name its snapshot cannot be repeated.
- **Roboflow credits are prepaid and finite** — 20 for the project's entire lifetime, with
  GPU training billed at 30 minutes per credit. Nothing here spends a credit without the
  estimate and approval that `docs/credit-budget.md` and `credit_gate.py` require.
"""
