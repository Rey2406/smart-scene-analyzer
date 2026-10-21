# AGENTS.md — Smart Scene Analyzer

<!--
  This file is Codex's persistent constraint layer. Codex loads repository AGENTS.md
  files into sessions, so project-wide rules belong here — not in individual prompts.

  Lesson 01 walks you through filling in the TODO markers. Delete this comment
  when you do.

  `CLAUDE.md` is the Claude Code-facing mirror of this file: same project rules, with
  the role table pointing at subagents instead of Skills and a hooks section describing
  enforcement Codex does not get. A project rule changed here and not there means two
  assistants working in one repo under different rules — change both.
-->

## Project

The Smart Scene Analyzer is a mobile perception system that runs entirely on the handset.
An Expo app for iOS and Android captures or picks an image, runs object detection and
monocular depth estimation **on the device**, fuses them, and draws the result. No image
leaves the phone, and nothing on the inference path touches a network.

- **Detection / classification:** YOLO11, exported to int8 TFLite
- **Monocular depth:** Depth Anything V2, exported to ExecuTorch (`.pte`)
- **On-device runtimes:** `react-native-fast-tflite` (detection), `react-native-executorch` (depth)
- **App:** Expo / React Native (TypeScript), iOS and Android from one codebase
- **Training / export / reference implementation:** Python in `src/`
- **Experiment tracking / registry:** MLflow
- **Dataset versioning:** Roboflow

## Architecture

```
   Expo dev build  (app/)                    iOS · Android
      │  capture or pick an image  (SOURCE space)
      ├──────────────────────┬──────────────────────┐
      ▼                                             ▼
   letterbox · normalize             letterbox · normalize
   → DET_INPUT space                 → DEPTH_INPUT space
   (its own scale + offset)          (its own scale + offset)
      │        ← the artifact contract →            │
      ▼                                             ▼
   YOLO11 int8 .tflite               Depth Anything V2 .pte
   (react-native-fast-tflite)        (ExecutorchModule)
      │  decode + NMS in TS                         │  relative inverse depth
      │  un-letterbox → SOURCE px                   │  (stays in DEPTH_INPUT)
      └──────────────────────┬──────────────────────┘
                             ▼
              Scene Understanding Layer   (TypeScript, pure)
                             ▼
              boxes + relative depth, drawn on screen

   ── no network call anywhere on this path ──

   src/  (Python)  trains the models, exports the artifacts above,
                   and remains the reference implementation the
                   device is checked against.
```

**There are two letterbox transforms, not one.** The two models take different input
sizes and dtypes, so each gets its own scale and offset, and there are two model-input
coordinate spaces rather than one shared one. A detection is un-letterboxed with the
*detection* transform; applying the depth one — or assuming a single shared transform —
returns boxes that are wrong by a scale factor and still comfortably in range.

Full detail: `docs/architecture.md`. Decisions and their rationale:
`docs/decisions/`. The move from a client/server split to this is recorded in the ADR
that superseded the cloud inference target — read it before proposing anything that puts
a server back on the request path.

## Layout

| Path | Contents |
|---|---|
| `src/smart_scene_analyzer/` | Python: training, export, and the **reference implementation** |
| `app/` | The Expo app. TypeScript. This is the product |
| `app/assets/models/` | The bundled `.tflite` and `.pte` artifacts. **What is here is what runs** |
| `app/src/fusion/` | The TypeScript port of `src/`'s fusion layer. Must agree with it |
| `tests/` | Unit, integration, and regression tests — including export parity |
| `docs/` | Architecture, roadmap, model cards |
| `docs/decisions/` | ADRs — one file per architectural decision |
| `docs/credit-budget.md` | The Roboflow credit ledger. **Read it before any billed call.** |
| `docs/artifact-budget.md` | Model and app size. The budget that fails at install time |
| `data/` | Dataset exports. **Gitignored.** Versioned in Roboflow. |
| `notebooks/` | Exploration only. Nothing in a notebook is production code. |
| `.agents/plugins/` | Local Codex marketplace manifest |
| `plugins/smart-scene-analyzer/` | Local Codex role Skills plugin |
| `.claude/agents/` | Claude Code subagent definitions |
| `.claude/skills/` | This project's own procedures |
| `.claude/hooks/` | Enforcement scripts — Claude Code only; see the note below |

**The table above is the target layout. Lesson 04 finished the Python side**:
`src/smart_scene_analyzer/` is a real package (training, export, fusion, geometry,
inference, preprocess), `tests/` covers it, and `docs/architecture.md`, `docs/taxonomy.md`,
and ADRs 0001–0004 all exist. **Lesson 05 is scaffolding the mobile side.** `app/` is a
real Expo development build — both ML runtimes installed, `app/assets/models/` holds the
real exported artifacts — but `app/src/` (the TypeScript fusion port,
`inference/detector.ts`, `inference/depth.ts`, `geometry/toScreen.ts`, the Photo/Live tabs)
does not exist yet. Still genuinely missing: `plugins/smart-scene-analyzer/`, which the
marketplace manifest points at — the Codex plugin install in `README.md` cannot succeed
until it is built. Do not treat either absence as breakage; each lands where this table
says when its lesson builds it.

## Engineering roles

Codex roles are reusable Skills supplied by the local `smart-scene-analyzer` plugin.
Install the plugin using the instructions in `README.md`, then use the relevant Skill
for work in its ownership area.

| Role | Owns | Skill |
|---|---|---|
| Architecture | System design, folder structure, engineering plan | `architecture` |
| Documentation | README, architecture docs, contribution guide, model cards | `documentation` |
| DevOps | Dev environment, mobile builds, GitHub Actions, releases | `devops` |
| Dataset Engineer | Roboflow projects, annotation review, versioning, quality analysis | `dataset-engineer` |
| Data Pipeline | Preprocessing, augmentation, conversion, validation scripts | `data-pipeline` |
| ML Engineer | Training, fine-tuning, MLflow, **export, quantization, parity** | `ml-engineer` |
| Evaluation | Metrics, confusion matrices, run comparison, model cards | `evaluation` |
| QA | Unit, integration, and regression tests | `qa` |
| Integration | Fusing detections with depth into one scene result | `integration` |
| Mobile | The Expo app: on-device inference, fusion, overlays, capture, latency | `mobile` |

Roles are added as the course progresses (ML Engineer, QA, Integration in
Lessons 03–04; Mobile in Lesson 05).

Two boundaries are load-bearing rather than tidy: **Evaluation does not edit code** — an
agent that can adjust the model when it dislikes the number is not measuring anything —
and **Mobile does not change the exported artifacts, while ML Engineer does not change the
app**. When the device disagrees with the reference implementation, that disagreement is
the finding. One role owning both sides would resolve it by adjusting whichever side was
easier to reach, and *which* of the two was wrong would stop being knowable.

## Project procedures

`.claude/skills/` holds this project's own procedures, as distinct from vendor knowledge.
They are written as Claude Code Skills, but the content is platform-neutral — read them
directly and follow them.

| Procedure | Use it when |
|---|---|
| `credit-ledger` | Before and after **any** operation that spends Roboflow credits |
| `error-triage` | Diagnosing weak classes — the DATA / TAXONOMY / MODEL classification |
| `annotation-conversion` | Converting any third-party annotation format to YOLO |
| `dataset-qa-sweep` | After every upload, before every version generation |
| `offline-suite` | Writing or reviewing tests around a model, a paid API, or a GPU |

## Enforcement you do not get

`.claude/hooks/` blocks two things automatically in Claude Code: a billed Roboflow call
with no cost estimate recorded in the ledger, and any write to `src/` **or `app/`**
containing a metric-depth identifier (`depth_meters`, `distance_cm`, `to_metres`) — in
Python, TypeScript, or TSX.

The second one reaches `app/` because that is where depth is computed now. When the models
ran on a server, the phone only displayed a number somebody else had produced; guarding the
Python was enough. It is not any more.

**Codex does not run these hooks.** Working here, those two rules are yours to keep
rather than the harness's to enforce, which makes them worth more attention, not less.
Read `.claude/hooks/credit_gate.py` and `units_guard.py` — they are short, and they state
the rule more precisely than prose can. In summary, what they check:

- **The credit gate** requires a row in the `docs/credit-budget.md` Ledger with a numeric
  **Estimated** and a blank **Actual**, not exceeding **Remaining**, and refuses outright
  below a **0.5-credit reserve floor**. Keep the file's table shape — the parser depends
  on it.
- **The units guard** matches the *text* of the write, not just identifiers. A bare
  `metres`/`meters` in a comment, docstring, or UI string is caught too, which is
  intended: the claim is as wrong in prose as in a variable name. Use `relative_depth`,
  `relative_inverse_depth`, or `depth_score`.

## Conventions

### Python

- Python 3.11+, managed with `uv`. Never `pip install` into the system interpreter.
- Formatting and linting: `ruff`. Type checking: `mypy` on `src/`.
- Public functions carry type hints and a docstring stating units and shapes.
  Ambiguity about whether a depth value is metres or normalized disparity is a real
  source of bugs in this project — say which.

### TypeScript (`app/`)

- Expo SDK with TypeScript. `npm` for the app; `uv` never touches `app/`.
- **Node 22.13+.** The current Expo SDK/React Native toolchain (Metro, the RN CLI)
  requires it — `npm install` warns loudly (`EBADENGINE`) under Node 20, and `expo
  start`/`expo run:*` print their own warning too. `nvm` is the easy way to manage this
  alongside other projects' Node versions.
- **This is a development build, not Expo Go.** Both ML runtimes are native modules, so
  the app cannot run in the Expo Go sandbox.
- **`app/ios/` and `app/android/` are generated and gitignored.** `expo prebuild` writes
  them, and the next prebuild discards whatever anyone hand-edited there. Native
  configuration — permissions, plugins, build settings — belongs in `app.json` or a config
  plugin, which is the input that survives.
- `app/src/fusion/` is a **port**, not an original. It must reproduce
  `src/smart_scene_analyzer/fusion.py` on the shared fixtures. When they disagree, fix the
  port or record why the reference is wrong — never adjust the fixtures to agree.
- Tensor code states its layout. `[1, 3, H, W]` float32 NCHW is not interchangeable with
  `[1, H, W, 3]` uint8 NHWC, and a silent mismatch produces plausible garbage rather than
  an error.

### Commands

```bash
uv sync                  # base + dev dependencies
uv sync --extra dataset --extra ml --extra depth   # add a lesson's dependencies

uv run pytest                                   # run tests, including export parity
uv run pytest -m "not integration"              # skip anything needing weights on disk
uv run pytest tests/test_units_guard.py         # one file
uv run pytest tests/test_units_guard.py::test_blocks_metric_depth   # one test
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy src          # type check — strict

cd app && npm install          # install app dependencies
cd app && npx expo run:ios     # build + install to the iOS Simulator
cd app && npx expo run:android # build + install to the Android Emulator
cd app && npx tsc --noEmit     # type check the app
```

`uv sync` installs only the base dependencies. The model, dataset, and MLOps packages are
**extras** — `dataset`, `ml`, `depth` in `pyproject.toml` — so each arrives in the lesson
that needs it. `uv sync --extra <name>` syncs the environment to exactly what you name;
pass every extra you still want.

Tests that need a checkpoint or an exported artifact carry the `integration` marker, so
the default suite runs with no GPU, no network, and no weights on disk. A new test that
needs any of those is either mocked or marked — see `.claude/skills/offline-suite`.

The first `expo run:*` compiles native code and takes minutes. After that, TypeScript
edits hot-reload; only a native dependency change requires another build.

### Secrets

API keys are read from the environment, never hardcoded and never committed.
`.env` is gitignored; `.env.example` documents the required variables.

Required: `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `MLFLOW_TRACKING_URI` — all
**dev-time only**. None is a runtime secret, because there is no runtime service.

**The app ships with no credentials of any kind.** Anything prefixed `EXPO_PUBLIC_` is
embedded in the bundle and readable by anyone who installs it, and this app has nothing to
authenticate to.

### Data and models

- Do not commit images, dataset exports, or model weights.
- Datasets are versioned in Roboflow; a dataset version is an immutable snapshot.
  Reference it by version number in code and docs, never as "the latest".
- Models and metrics are tracked in MLflow. A model referenced in code carries an
  explicit registry version.

### Testing

- Every module in `src/` has a matching test file.
- Model inference is mocked in unit tests. Tests must run without a GPU, without network
  access, and without weights on disk.
- **Export parity is a test, not a manual check.** The exported artifact and the PyTorch
  reference must agree on the same fixtures, within a stated tolerance. An export nobody
  compared against the reference is not known to be correct — only known to run.
- The TypeScript fusion suite runs the **same fixtures** as the Python one. That shared
  fixture set is the only thing keeping two implementations of one algorithm honest.

## Credit budget

**This project has a hard budget of 20 Roboflow credits for its entire lifetime.** The
budget is not a guideline. It covers Lessons 01 through 06, and there is no top-up.

The running ledger is `docs/credit-budget.md`. It is the source of truth for what has
been spent and what remains.

### Before any billed call

State three things and wait for approval:

1. The operation and the rate that applies.
2. The estimated cost, with the arithmetic shown.
3. The remaining balance from `docs/credit-budget.md`.

If the estimate exceeds the remaining balance, **stop and report**. Do not run a smaller
version of the operation to fit — say what it would cost and let the user decide what to
cut.

After the call completes, append the actual cost to the ledger. Estimated and actual
diverging is information worth having, not an error to hide.

### What one credit buys

Rates change upstream. Verify against Roboflow's current published rates rather than
trusting this table, which is recorded here so the order of magnitude is visible.

| Operation | 1 credit |
|---|---|
| Uploads | 10,000 images |
| Storage | 5,000 images/month |
| Version generation | 20,000 images |
| Auto Label | **100 images** |
| Model training (GPU) | **30 minutes** |
| Hosted serverless inference | 500 seconds of execution |
| Self-hosted inference | 3,000 images — **metered, not free** |

Free: Auto Label's 4-image "Generate Test Results" preview, Roboflow Instant training,
Universe search, RoboQL queries, tags, splits, and workflow authoring and validation.

### Forbidden without explicit approval

| Operation | Why |
|---|---|
| **RF-DETR NAS** (`rfdetr-nas-*-parent`) | Roboflow's *own guidance recommends this by default.* A NAS run trains dozens of child models over hours at 2 credits/hour, and fails outright on non-Core plans. This project overrides that recommendation. |
| Dedicated deployments | Bills uptime, not usage. One deployment left running is 24 credits/day — more than the whole budget. |
| Batch processing on GPU | 4 credits/hour, the worst rate available. |
| Auto Label beyond 100 images | Linear at 0.01 credits/image. 2,000 images is the entire budget. |
| `datasource_trigger`, `connect_cloud_storage` | Mirror runs bill, and `trigger` defaults to **true**. |
| Video streams and WebRTC, hosted or local | Self-hosted video alone caps at 20 credits/month. |

## Artifact budget

`docs/artifact-budget.md` is the second ledger, and it bills on a different principle
again. Roboflow credits are prepaid and run out; the artifact budget does not bill at all.
It **fails**, at build or install time, and the failure is not gradual.

A model file that is 40 MB too large does not cost 40 MB of money. It pushes the app past
a store's over-the-air download threshold, or past what a mid-range device will hold in
memory with two runtimes loaded. There is no invoice and no alert — the first signal is a
build failure or a crash on somebody else's phone.

Record every artifact's size **before** bundling it, and the install size per platform
after. This project ships two native ML runtimes, which makes it real rather than
theoretical.

**Shipping both is not yet a settled decision.** `docs/artifact-budget.md` carries two
`⚠️ OPEN` items that reshape the app rather than tune it: the device floor has not been
chosen (the runtimes' own floor is iOS 17+ / Android 13+, and the unmeasured memory
ceiling will push it up), and whether two native runtimes are affordable at all is
unverified. The fallback is consolidating onto one runtime — YOLO11 to `.pte`, or Depth
Anything V2 to TFLite. Do not write prose or code treating the split as final until
somebody has measured it.

## The artifact contract

The app and the export pipeline share the model files, and that is all they share.

- **The export defines it; the app consumes it.** Input tensor shape and dtype, output
  tensor count and order, class label order, and normalization constants are fixed by the
  export and assumed by the app. Nothing checks them at runtime. An app built against last
  week's label order runs, draws boxes, and names them wrong.
- **`app/assets/models/` is what runs** — not what the export script would produce if you
  ran it now.
- **The artifacts are gitignored.** `.gitignore` excludes `*.tflite` and `*.pte`, so the
  files the app actually runs are not in version control and a fresh clone cannot build
  the app. They are produced by the export step and reproduced from MLflow — which is why
  a model referenced in code carries an explicit registry version, and why the export
  parity test is the only thing that says a rebuilt artifact is the same artifact. If the
  models are missing, that is expected; re-export rather than hunting for them in git.
- **Three coordinate spaces, all named:** source image pixels, letterboxed model-input
  pixels (with their own scale and offset), and screen points. Converting between them
  belongs in one named function with both spaces in its signature.
- **Depth is `relativeDepth`**: relative inverse depth, larger is nearer, no unit and no
  scale, comparable only within one image. The UI may order and shade by it; it may not
  print a distance.
- **Zero detections is a successful result with an empty list**, not an error. It must
  look different on screen from "still loading" and from "the model failed to load".

## Constraints for assistants

- **Ask before installing a new dependency.** This project has a deliberately small
  surface. In `app/`, an addition is also **bytes on somebody's phone** — check what the
  Expo SDK already bundles, and check `docs/artifact-budget.md` before adding anything
  that carries native code.
- **Estimate credits before spending them**, per the Credit budget section above. An
  assistant that spends the budget discovering what something costs has spent it.
- **Do not modify `data/`.** Dataset changes go through Roboflow, so that the change
  is versioned and reviewable.
- **Do not commit or push unless asked.**
- **State units and coordinate conventions** when writing geometry or depth code.
  Bounding boxes are `xyxy` in absolute pixels of a **named** space — source image, model
  input, or screen — unless a function signature says otherwise.
- **Do not put a network call on the inference path.** Not for a model, not for a label
  set, not for telemetry that blocks a result. That the phone works on a plane is a
  property of this architecture, and it is one line of code away from being lost.

## Decisions on record

- **Inference target: on-device.** Both models run on the handset; no network call exists
  on the inference path. Rejected: a hosted detection API (bills per request, and a camera
  client makes far more of them than a benchmark does) and a self-hosted container
  (transit alone can consume the whole 400 ms p95 budget on a link we do not control). The
  cost of the choice is an export toolchain, two native runtimes, per-platform numerical
  parity, and an app-store release per model update — recorded, accepted, and the reason
  most of this file exists. See
  [`docs/decisions/0001-inference-target.md`](docs/decisions/0001-inference-target.md).
- **Detection architecture: YOLO11 over RF-DETR.** Decided on export maturity onto the
  on-device target ADR 0001 already committed to — Ultralytics' int8 TFLite export path
  is proven; RF-DETR's mobile export path is not, and its recommended training route
  (NAS) is separately forbidden on credit-budget grounds. See
  [`docs/decisions/0004-detection-architecture-yolo11-over-rf-detr.md`](docs/decisions/0004-detection-architecture-yolo11-over-rf-detr.md).
- Requirements, with the numbers this design is held to:
  [`docs/requirements.md`](docs/requirements.md).

<!-- TODO(Lesson 01): record the MLflow hosting decision and reflect it here. -->

## Maintenance

Keep this file semantically aligned with `CLAUDE.md`. Adapt platform-specific wording,
but preserve all project constraints in both files.
