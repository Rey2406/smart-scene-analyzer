"""N20 — export parity: the artifact against the reference, on shared fixtures.

**Single responsibility:** answer whether an exported artifact computes the same thing the
reference does, within a stated tolerance, and say where it does not.

An export nobody compared against the reference is not known to be correct. It is only
known to run. This module is the difference between those two statements, and it is a test
rather than a manual check because it is the only thing standing between a quantization
regression and an app-store submission.

**Both sides go through the same protocol.** The reference backend and an
artifact-backed backend both satisfy `DetectorBackend` / `DepthBackend`, so parity is a
loop over fixtures calling the same method twice. That is the payoff of the seam in
`inference.protocols` beyond mockability.

What is compared, per model:

- **Detection:** matched boxes within an IoU tolerance, per-box confidence within a
  tolerance, and the label of every matched pair. A label mismatch is a **hard failure,
  not a tolerance**, because it means the ordered label list in the manifest disagrees with
  the artifact — the failure mode that is invisible on a device.
- **Depth:** the map's values within a tolerance, **and** the pairwise ordering over the
  fixture's annotated object pairs. Ordering is the property the product depends on (N4),
  and an artifact could pass a value tolerance while inverting two nearby objects.

The tolerance is **stated**, not discovered. A tolerance widened until the test passed is a
description of the current artifact, not a gate on the next one.

**Integration-marked.** It needs both a checkpoint and an exported artifact on disk.

Planned public surface:

    class ParityReport      # per-fixture pass/fail with the worst observed deviation
    def check_detector_parity(
        reference: DetectorBackend,
        artifact: DetectorBackend,
        fixtures: Sequence[FusionFixture],
        tolerances: DetectorTolerances,
    ) -> ParityReport

    def check_depth_parity(
        reference: DepthBackend,
        artifact: DepthBackend,
        fixtures: Sequence[FusionFixture],
        tolerances: DepthTolerances,
    ) -> ParityReport
        \"\"\"Compares unit-less relative inverse depth values and, separately, the pairwise
        ordering the product actually relies on.\"\"\"
"""
