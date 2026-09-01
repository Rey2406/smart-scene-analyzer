"""The reference model backends and the pure post-processing around them.

**Single responsibility:** run the two reference models and turn the detector's raw output
into boxes — with the model-loading half quarantined behind protocols so the rest of the
package stays testable.

The package is deliberately split in two:

- **Impure and marked.** `detector.py` and `depth.py` load weights, import `ultralytics`
  and `transformers`, and are used only from tests carrying `@pytest.mark.integration`.
- **Pure and always tested.** `decode.py` and `nms.py` take arrays and return boxes. No
  weights, no GPU, no network.

`protocols.py` is the seam between them, and it is the seam the whole offline test suite
rests on. It is also the seam that the still-open "one runtime or two" decision turns:
consolidating both models onto a single runtime replaces the implementations behind these
protocols and changes no contract in `docs/architecture.md` §4 except the artifact
manifest.
"""
