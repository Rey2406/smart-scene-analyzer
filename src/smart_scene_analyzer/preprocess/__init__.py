"""Image entry and the two model-input tensors (pipeline stages S0, S1, S2).

**Single responsibility:** turn image bytes into exactly the tensors the two exported
artifacts expect, and hand back the transforms needed to get answers back out.

There are **two preprocessing paths, not one.** The sketch in `CLAUDE.md` shows a single
`letterbox · normalize` node fanning out to both models; that is a simplification this
package corrects. Detection wants `[1, 640, 640, 3]` NHWC at an export-determined dtype;
depth wants `[1, 3, 518, 518]` float32 NCHW with channel normalization. Different sizes
mean different letterbox transforms, which means two model-input coordinate spaces.
Collapsing them produces boxes that are wrong by a scale factor and still in range.

Everything about the tensors is **export-owned** (`docs/architecture.md` C3, C4). The
values assumed by this package are recorded as assumptions A2/A3/A4 and are authoritative
only in `model_contract.json`.
"""
