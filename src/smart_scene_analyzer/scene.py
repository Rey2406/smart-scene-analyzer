"""The end-to-end reference pipeline, and the owner of the degraded path (N12).

**Single responsibility:** wire S0 through S6 together in the order the device runs them,
and decide what happens when one branch fails.

This is the reference orchestration. The app has its own driver in TypeScript; this one
exists so that "what the pipeline should have produced" is a function anybody can call on
a workstation, which is what parity testing compares against.

**Backends are injected, never constructed here.** `analyse` takes a `DetectorBackend` and
an optional `DepthBackend` (`protocols.py`). Nothing in this module imports a runtime, so
a test drives the whole pipeline with two fixture-backed doubles and no weights on disk.

## Failure asymmetry (N12), which is a design constraint and not error handling

- **Depth unavailable** -> degrade to detection-only. `depth_map=None` flows into
  `fusion.fuse`, every `relative_depth` is `None`, and `SceneResult.depth_available` is
  `False`. The user sees boxes.
- **Detection unavailable** -> fail visibly. There is no partial result to render, and a
  scene with a depth map and no objects is indistinguishable on screen from a scene with
  nothing in it. Never render a partial result that looks complete.
- **Zero detections** -> a successful `SceneResult` with an empty list (F8). Distinct from
  both of the above, and the UI must show it as distinct.

## What is deliberately absent

No network call, no telemetry, no retry, no cache lookup. Not even a non-blocking one:
that would put a radio on a path this architecture guarantees is offline (ADR 0001, N1a,
N15, F10). If a future edit adds an import that opens a socket here, that is a reversal of
ADR 0001 and not an implementation detail.

Model **load** is not on this path either. Loading is a separate lifecycle owned by the
app's `ModelHost` and budgeted separately (N6, 3.0 s), because folding a three-second load
into a 400 ms budget hides it. The reference mirrors that split by taking already-built
backends as arguments.

Planned public surface:

    def analyse(
        data: bytes,
        detector: DetectorBackend,
        depth: DepthBackend | None,        # None == the N12 degraded path
        settings: Settings,
    ) -> SceneResult
        \"\"\"Decode -> preprocess (twice) -> infer (both) -> decode+NMS -> fuse.
        Boxes in the result are xyxy absolute pixels in SOURCE space; depth fields are
        unit-less relative inverse depth, larger is nearer.\"\"\"

    def build_depth_preview(depth_map: DepthMap) -> DepthPreview
        \"\"\"A rank-preserving remap to uint8 0..255 for shading (F9). For ordering and
        shading only; it is not a measurement and carries no unit.\"\"\"
"""
