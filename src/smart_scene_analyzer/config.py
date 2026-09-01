"""Dev-time settings, read from the environment.

**Single responsibility:** be the one place that reads `os.environ`, so that no other
module has to.

Everything here is **development-time only**. `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`,
and `MLFLOW_TRACKING_URI` are used by training, dataset, and export work on a workstation.
None of them is a runtime secret, because there is no runtime service (ADR 0001) and the
app ships with no credentials of any kind.

Also holds the pipeline constants that `docs/architecture.md` §11 records as
**assumptions** rather than requirements — `DECODE_LONG_EDGE` (A1), `DET_SIZE` (A2),
`DEPTH_SIZE` (A3), the letterbox pad value (A4), `INSET_FRACTION` (A5), `MAX_SAMPLES`
(A6), `MAX_DETECTIONS` (A7), and the confidence and NMS thresholds (A8). They live here
so that a change to one is a change to one line, and so that a reader can see which
numbers nobody has yet justified from a requirement.

The export-owned half of those values (tensor shapes, dtypes, label order, thresholds
shipped with the artifact) is authoritative in `model_contract.json`, not here.
`smart_scene_analyzer.export.contract` is what reads it. When the two disagree, the
manifest wins — it is the one describing the file that actually runs.

Planned public surface:

    class Settings(BaseSettings)         # env-backed; no default points at a network
    def load_settings() -> Settings      # cached
"""
