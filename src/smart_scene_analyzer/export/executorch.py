"""Depth Anything V2 to ExecuTorch `.pte`.

**Single responsibility:** produce the depth artifact the app bundles, and emit the half of
`model_contract.json` that describes it.

The vision transformer is the harder of the two artifacts to shrink, and it is the one that
dominates both N9a (40 MB combined) and N6 (the 1.4 s of the 3.0 s cold-load budget
allocated to deserializing it).

The exported graph must preserve the output's meaning as well as its shape: relative
inverse depth, larger nearer, no unit and no scale. An export that flips polarity or
rescales produces a map that passes a shape check and renders every scene inside out.
That is why the manifest carries an explicit `semantics` string the app asserts on at load
(`export.contract`).

**Integration-marked.** It writes files and needs a checkpoint.

Whether this exporter is the one that runs at all is an **open decision**: the other
consolidation branch replaces it with a TFLite export of Depth Anything V2, whose
unverified part is quantizing a transformer to int8 without unacceptable N4 loss
(`docs/artifact-budget.md`, architecture §10.4).

Planned public surface:

    def export_depth(
        checkpoint: Path,
        out_path: Path,
        depth_size: int,
    ) -> DepthArtifactSpec
        \"\"\"-> the manifest fragment for this artifact: input shape/dtype/layout,
        normalization constants, output shape, `semantics`, and sha256.\"\"\"
"""
