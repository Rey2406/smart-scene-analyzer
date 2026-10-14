# ADR 0004 — Detection architecture: YOLO11 over RF-DETR

- **Status:** accepted
- **Date:** 2026-10-07
- **Deciders:** Project team

## Context

ADR 0001 commits detection to run on-device, in int8 TFLite, via
`react-native-fast-tflite`. That commitment is the constraint this decision inherits:
whatever detection architecture is chosen has to actually clear a CNN-to-TFLite int8
export, on a toolchain this project can run without a GPU server, not merely perform well
in a PyTorch benchmark.

Roboflow's own training guidance (`roboflow:training-and-evaluation`) recommends RF-DETR
as a default. RF-DETR is a transformer-based (DETR-style) detector, and its mobile export
path — to TFLite in particular — is far less proven than the CNN export path Ultralytics
ships for YOLO11: quantization-aware int8 export, NMS baked into a known tensor layout,
and a direct `.pt` → `.tflite` command this project's `export/tflite.py` already exercises.
Choosing RF-DETR here would mean discovering that export path during Lesson 04, against
this project's fixed scope, rather than reusing one that already works.

This is a separate question from RF-DETR **NAS training**, which is forbidden outright on
credit-budget grounds (`docs/credit-budget.md`, `CLAUDE.md`): that rules out one *training
route*, not the architecture as a whole. Even setting the NAS cost aside, the export
maturity gap would still apply to any RF-DETR checkpoint, however it was trained.

## Options considered

| Option | Pros | Cons |
|---|---|---|
| **A — YOLO11** (Ultralytics) | Established `.pt` → int8 `.tflite` export, exercised by `src/smart_scene_analyzer/export/tflite.py`; fine-tunes locally with `ultralytics` at no Roboflow training cost; NHWC uint8 input and NMS layout well-documented for on-device decode | CNN architecture — generally behind transformer-based detectors on accuracy at comparable latency; smaller ecosystem of pretrained checkpoints than some alternatives |
| **B — RF-DETR** | Roboflow's recommended default; competitive accuracy; strong if deployed server-side or via ONNX/TensorRT | Mobile/TFLite export path is unproven for this toolchain; the model-recommended training route (NAS) is separately forbidden on credit-budget grounds; would require discovering or building a quantized TFLite export during an already-scoped lesson |

## Decision

**Detection uses YOLO11, exported to int8 TFLite via Ultralytics' export tooling** —
not RF-DETR. The deciding factor is export maturity onto the on-device target ADR 0001
already committed to, not accuracy in isolation.

## Consequences

**Easier:** the export pipeline has one well-exercised path from checkpoint to the
artifact the app bundles, with no export-toolchain research needed inside the lesson
schedule. Local fine-tuning with `ultralytics` costs zero Roboflow credits.

**Harder / foreclosed:** this project does not get RF-DETR's accuracy ceiling, and does
not benefit from Roboflow's own default recommendation or its model-eval tooling tuned
around it.

**What now depends on this decision:**

- `src/smart_scene_analyzer/inference/detector.py` — the reference YOLO11 backend
- `src/smart_scene_analyzer/export/tflite.py` — YOLO11 → int8 `.tflite`
- `src/smart_scene_analyzer/training/train_detector.py` and `scripts/train_detector.py` —
  the YOLO11 fine-tuning entry points
- `pyproject.toml` — the `ultralytics>=8.3` dependency
- `docs/artifact-budget.md` — sizes recorded for `yolo11n` specifically
- `docs/architecture.md` and `AGENTS.md` / `CLAUDE.md` — diagrams and tables naming YOLO11
  as the detection model
- `.claude/agents/ml-engineer.md`, `.claude/agents/dataset-engineer.md`,
  `.claude/skills/credit-ledger/SKILL.md` — the standing rule against RF-DETR NAS training,
  which remains in force independently of this ADR

**What this does not foreclose:** `src/smart_scene_analyzer/inference/protocols.py`
defines `DetectorBackend` as a structural protocol — arrays in, arrays out, no file paths
or sessions in the signature. A future detector swap, RF-DETR or otherwise, replaces the
implementation behind that protocol and leaves the data contract intact, the same seam
the "one runtime or two" consolidation decision already relies on.

## Revisit when

- Ultralytics or a third party ships a validated int8 TFLite export path for RF-DETR (or
  an equivalent transformer detector) with on-device latency and size comparable to
  YOLO11n, closing the gap this ADR is based on.
- Evaluation (`.claude/agents/evaluation.md`) finds YOLO11's accuracy insufficient against
  `docs/requirements.md`'s detection targets, and the shortfall survives data and taxonomy
  fixes (see `error-triage`) — i.e., the DATA/TAXONOMY causes are ruled out first.
- The credit-budget prohibition on RF-DETR NAS training is lifted or becomes moot (e.g. a
  non-NAS RF-DETR training route becomes available), **and** the export-maturity gap above
  has independently closed. Either alone is not sufficient.
