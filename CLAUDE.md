# CLAUDE.md — Smart Scene Analyzer

<!--
  This file is the harness's persistent constraint layer. It is loaded into every
  Claude Code session in this repo, so it is where project-wide rules belong —
  not in individual prompts, which are forgotten the moment the session ends.

  Lesson 01 walks you through filling in the TODO markers. Delete this comment
  when you do.

  `AGENTS.md` is the Codex-facing mirror of this file: same project rules, with the
  role table pointing at Skills instead of subagents and the hooks section saying
  Codex does not run them. A project rule changed here and not there means two
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
`docs/decisions/`. The move from a client/server split to this is
recorded in the ADR that superseded the cloud inference target — read it before
proposing anything that puts a server back on the request path.

## Layout

| Path | Contents |
|---|---|
| `src/smart_scene_analyzer/` | Python: training, export, and the **reference implementation** |
| `app/` | The Expo app. TypeScript. This is the product |
| `app/assets/models/` | The bundled `.tflite` and `.pte` artifacts. **What is here is what runs** |
| `app/src/fusion/` | The TypeScript port of `src/`'s fusion layer. Must agree with it |
| `scripts/` | Operational helpers — converters, dataset assembly, training entry point, verification. Not importable production code; that lives in `src/` |
| `tests/` | Unit, integration, and regression tests — including export parity |
| `docs/` | Architecture, roadmap, model cards |
| `docs/decisions/` | ADRs — one file per architectural decision |
| `docs/credit-budget.md` | The Roboflow credit ledger. **Read it before any billed call.** |
| `docs/artifact-budget.md` | Model and app size. The budget that fails at install time |
| `data/` | Dataset exports. **Gitignored.** Versioned in Roboflow. |
| `notebooks/` | Exploration only. Nothing in a notebook is production code. |
| `.claude/agents/` | Subagent definitions — the engineering roles |
| `.claude/skills/` | This project's own procedures. Read by role and by you |
| `.claude/hooks/` | Enforcement. These run whether or not anyone reads them |

**The table above is the target layout, not what is on disk today.** This repo is the
Lesson 01 scaffold: the harness is complete — `.claude/`, the two ledgers, the ADR
template, `pyproject.toml` — and the system it constrains is not yet written.
`src/smart_scene_analyzer/` holds only `__init__.py`, `app/` is an empty directory with a
README explaining why, `tests/` holds only the units-guard test, and
`docs/architecture.md`, `docs/taxonomy.md`, and the ADRs referenced elsewhere in this file
do not exist yet. Do not go looking for them, and do not treat their absence as breakage —
each is created by the lesson that needs it. When you create one, it lands where this
table says.

## Engineering roles

Work is delegated to specialized subagents in `.claude/agents/`. Each owns a slice of
the system and is invoked for work inside that slice.

| Role | Owns | Definition |
|---|---|---|
| Architecture | System design, folder structure, engineering plan | `.claude/agents/architecture.md` |
| Documentation | README, architecture docs, contribution guide, model cards | `.claude/agents/documentation.md` |
| DevOps | Dev environment, mobile builds, GitHub Actions, releases | `.claude/agents/devops.md` |
| Dataset Engineer | Roboflow projects, annotation review, versioning, quality analysis | `.claude/agents/dataset-engineer.md` |
| Data Pipeline | Preprocessing, augmentation, conversion, validation scripts | `.claude/agents/data-pipeline.md` |
| ML Engineer | Training, fine-tuning, hyperparameters, MLflow, run diagnosis | `.claude/agents/ml-engineer.md` |
| Evaluation | Metrics, confusion matrices, run comparison, model cards | `.claude/agents/evaluation.md` |
| Export | Export, quantization, and numerical parity against the reference | `.claude/agents/ml-engineer.md` |
| QA | Unit, integration, and regression tests | `.claude/agents/qa.md` |
| Integration | Fusing detections with depth into one scene result | `.claude/agents/integration.md` |
| Mobile | The Expo app: on-device inference, fusion, overlays, capture, latency | `.claude/agents/mobile.md` |

All roles ship with the scaffold, but each is introduced by the lesson that first needs
it — the Dataset Engineer in Lesson 02, the ML Engineer and Evaluation in Lesson 03,
Export, QA, and Integration in Lesson 04, Mobile in Lesson 05. Reading a role's
constraints before invoking it is part of the work.

Two boundaries are load-bearing rather than tidy:

- **Evaluation has no `Edit` tool.** It reports on models; it cannot change the code that
  produces them. An agent that can adjust the model when it dislikes the number is not
  measuring anything.
- **Mobile does not change the exported artifacts, and Export does not change the app.**
  When the device disagrees with the reference implementation, the disagreement is the
  finding. A single role that owned both sides would resolve it by adjusting whichever
  side was easier to reach, and the information — *which* of the two is wrong — would
  disappear without anyone deciding anything.

## Project skills

`.claude/skills/` holds this project's own procedures, as distinct from the vendor
knowledge in the `roboflow:*` skills. Invoke them by name.

| Skill | Use it when |
|---|---|
| `credit-ledger` | Before and after **any** operation that spends Roboflow credits |
| `error-triage` | Diagnosing weak classes — the DATA / TAXONOMY / MODEL classification |
| `annotation-conversion` | Converting any third-party annotation format to YOLO |
| `dataset-qa-sweep` | After every upload, before every version generation |
| `offline-suite` | Writing or reviewing tests around a model, a paid API, or a GPU |

A vendor skill encodes the vendor's defaults. These encode ours, and where they conflict,
ours wins — it is the one that knows the budget.

## Hooks

`.claude/hooks/` is the layer that does not rely on anyone reading anything. Configured in
`.claude/settings.json`.

| Hook | Fires on | Effect |
|---|---|---|
| `session_balance.py` | Session start | Puts the remaining credit balance into context |
| `notify_done.py` | The turn ending, or a prompt waiting on you | Desktop notification for the operator. Never blocks |
| `format_python.py` | After `Write`/`Edit` | `ruff format` + `ruff check --fix`. Never blocks |
| `credit_gate.py` | Before billed Roboflow MCP tools | **Blocks** unless the ledger holds a pending estimate that fits the balance |
| `units_guard.py` | Before `Write`/`Edit` to `src/` **or `app/`** | **Blocks** any metric-depth identifier, in Python or TypeScript |
| `artifact_drift.py` | After editing an export config or the taxonomy | Warns that the bundled model artifacts are older than what produced them |

The two that block are the point. `CLAUDE.md` *asks* for an estimate before spending and
*asks* that depth never be called metres; a permission prompt then offers a button that
says yes. Neither survives a confident model at the wrong moment. A hook returning exit 2
does.

What each of the two actually checks, because "satisfy it" needs a target:

- **`credit_gate.py`** reads `docs/credit-budget.md` and requires a Ledger row with a
  numeric **Estimated** and a blank **Actual** — a pending row — that does not exceed
  **Remaining**. It also refuses outright below a **0.5-credit reserve floor**, so the
  last lesson stays reachable. The hook parses the `| **Remaining** | <number> |` row and
  the Ledger table by shape: change that file's shape and the parser finds nothing, reads
  a balance of zero, and blocks everything.
- **`units_guard.py`** matches the *text* of any `.py`, `.ts`, or `.tsx` write under
  `src/` or `app/` — not just identifiers. A bare `metres`/`meters` in a comment,
  docstring, or UI string blocks the write too, which is intended: the claim is as wrong
  in prose as in a variable name. `node_modules`, `.venv`, `build`, and `dist` are exempt.
  Use `relative_depth`, `relative_inverse_depth`, or `depth_score`.

If a hook blocks you, the fix is to satisfy it — record the estimate, rename the field.
Do not edit the hook to get past it, and do not work around it by writing the file some
other way. If a hook is genuinely wrong, say so and stop; that is an ADR, not a patch.
`.claude/commands/adr.md` (`/adr`) writes one into `docs/decisions/`.

## Conventions

### Python

- Python 3.11+, managed with `uv`. Never `pip install` into the system interpreter.
- Formatting and linting: `ruff`. Type checking: `mypy` on `src/`.
- Public functions carry type hints and a docstring stating units and shapes.
  Ambiguity about whether a depth value is metres or normalized disparity is a real
  source of bugs in this project — say which.

### TypeScript (`app/`)

- Expo SDK with TypeScript. `npm` for the app; `uv` never touches `app/`.
- **This is a development build, not Expo Go.** Both ML runtimes are native modules, so
  the app cannot run in the Expo Go sandbox. `npx expo run:ios` / `run:android` build it.
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

cd app && npm install         # install app dependencies
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
**dev-time only**. None of them is a runtime secret, because there is no runtime service.

**The app ships with no credentials of any kind.** If you find yourself adding an
`EXPO_PUBLIC_` variable holding anything but a build flag, stop: anything so prefixed is
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
  compared against the reference is not known to be correct — it is only known to run.
- The TypeScript fusion suite runs the **same fixtures** as the Python one. That shared
  fixture set is the only thing keeping two implementations of one algorithm honest.

## Credit budget

**This project has a hard budget of 20 Roboflow credits for its entire lifetime.** The
budget is not a guideline. It covers Lessons 01 through 06, and there is no top-up.

The running ledger is `docs/credit-budget.md`. It is the source of truth for what has
been spent and what remains.

### Before any billed call

State three things and wait for approval:

1. The operation and the rate that applies (from `roboflow:plans-and-pricing`).
2. The estimated cost, with the arithmetic shown.
3. The remaining balance from `docs/credit-budget.md`.

If the estimate exceeds the remaining balance, **stop and report**. Do not run a smaller
version of the operation to fit — say what it would cost and let the user decide what to
cut.

After the call completes, append the actual cost to the ledger. Estimated and actual
diverging is information worth having, not an error to hide.

### What one credit buys

Rates change upstream. Verify against `roboflow:plans-and-pricing` rather than trusting
this table, which is recorded here so the order of magnitude is visible at a glance.

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
| **RF-DETR NAS** (`rfdetr-nas-*-parent`) | Roboflow's *own skill recommends this by default.* A NAS run trains dozens of child models over hours at 2 credits/hour, and fails outright on non-Core plans. This project overrides that recommendation. |
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
memory with two runtimes loaded, and the result is an app that some users simply cannot
install or run. There is no invoice and no alert — the first signal is a build failure or
a crash on somebody else's phone.

Record every artifact's size in `docs/artifact-budget.md` **before** bundling it, and the
install size per platform after. The two runtimes in this project make this real rather
than theoretical: `react-native-fast-tflite` and `react-native-executorch` each carry
native libraries, and we ship both.

**Shipping both is not yet a settled decision.** `docs/artifact-budget.md` carries two
`⚠️ OPEN` items, and they are the kind that reshape the app rather than tune it: the
device floor has not been chosen (the runtimes' own floor is iOS 17+ / Android 13+, and
the unmeasured memory ceiling will push it up), and whether two native runtimes are
affordable at all is unverified. If the combined native library size proves unacceptable,
the fallback is consolidating onto one runtime — YOLO11 to `.pte`, or Depth Anything V2 to
TFLite. Both are real workstreams. Do not write prose or code that treats the two-runtime
split as final until somebody has measured it.

## The artifact contract

The app and the export pipeline share the model files, and the model files are the only
thing they share. This is where a change breaks something nobody is looking at.

- **The export defines it; the app consumes it.** Input tensor shape and dtype, output
  tensor count and order, class label order, and normalization constants are all fixed by
  the export and assumed by the app. Nothing checks them at runtime. An app built against
  last week's label order runs, draws boxes, and names them wrong.
- **`app/assets/models/` is what runs.** Not what the export script would produce if you
  ran it. `artifact_drift.py` warns when the two have diverged; it cannot know whether the
  difference matters, so it tells you rather than deciding for you.
- **The artifacts are gitignored.** `.gitignore` excludes `*.tflite` and `*.pte`, so the
  files the app actually runs are not in version control and a fresh clone cannot build
  the app. They are produced by the export step and reproduced from MLflow — which is why
  a model referenced in code carries an explicit registry version, and why the export
  parity test is the only thing that says a rebuilt artifact is the same artifact. If the
  models are missing, that is expected; re-export rather than hunting for them in git.
- **Coordinate spaces are named, not assumed.** There are three: source image pixels,
  letterboxed model-input pixels (with their own scale and offset), and screen points.
  Converting between them belongs in one named function with both spaces in its signature.
  A box indexed into the wrong space returns a number — the wrong number, in range.
- **Depth is `relativeDepth`**: relative inverse depth, larger is nearer, no unit and no
  scale, comparable only within one image. The UI may order and shade by it. It may not
  print a distance. `units_guard` now enforces this in `app/` as well as `src/`, because
  `app/` is where the number is computed.
- **Zero detections is a successful result with an empty list**, not an error. It must
  look different on screen from "still loading" and from "the model failed to load".

## Constraints for assistants

- **Ask before installing a new dependency.** This project has a deliberately small
  surface. In `app/`, an addition is also **bytes on somebody's phone** — check what the
  Expo SDK already bundles, and check `docs/artifact-budget.md` before adding anything
  that carries native code.
- **Estimate credits before spending them**, per the Credit budget section above. An
  assistant that spends the budget discovering what something costs has spent it. The
  `credit_gate` hook will stop you if you skip it.
- **Do not modify `data/`.** Dataset changes go through Roboflow, so that the change
  is versioned and reviewable.
- **Do not commit or push unless asked.**
- **State units and coordinate conventions** when writing geometry or depth code.
  Bounding boxes in this project are `xyxy` in absolute pixels of a **named** space —
  source image, model input, or screen — unless a function signature says otherwise.
- **Do not put a network call on the inference path.** Not for a model, not for a label
  set, not for telemetry that blocks a result. That the phone works on a plane is a
  property of this architecture, and it is one line of code away from being lost.
- **Never let a hook be the thing you edit.** If one blocks you, satisfy it or stop.

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
