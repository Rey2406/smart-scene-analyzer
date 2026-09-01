"""The artifact boundary: producing the files the app bundles, and proving they are right.

**Single responsibility:** define the artifact contract and prove the exported artifacts
satisfy it — because this is the one seam where a mistake ships to a user.

`docs/architecture.md` §1 names the asymmetry this package lives on. `src/` ships on a
`git push` and can be fixed in minutes; `app/` ships through an app-store review and
cannot. So everything provable on this side of the boundary is proved here, on fixtures,
before it is allowed to cross.

Three things cross: a `.tflite`, a `.pte`, and `model_contract.json`. The two model files
are gitignored, so a fresh clone cannot build the app until somebody re-exports — that is
expected, not breakage. The manifest is not gitignored, because it is the only part of the
contract a reviewer can read.

**Nothing checks the contract at run time.** An app built against last week's label order
runs, draws boxes, and names them wrong. `parity.py` and `artifact_drift.py` are the
mitigation, and neither is automatic in the way a type checker is — which is why N20 is a
quality gate.

Whether both artifacts target two runtimes or one is an **open decision**
(`docs/artifact-budget.md`). This package's structure does not depend on the answer:
consolidation changes which exporter module runs, not what the manifest has to state.
"""
