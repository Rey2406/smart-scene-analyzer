# Smart Scene Analyzer

A mobile perception app. On iOS or Android, you capture a photo or pick one from your
library, and the app runs object detection and monocular depth estimation **on the
handset itself** — no image, and no network call of any kind, leaves the phone. The
result is drawn on screen as bounding boxes with class labels and a per-object relative
depth score (nearer/farther, not a physical distance). This is not a client for a
server: there is no server, and the app ships with no credentials to reach one.

| Layer | Choice |
|---|---|
| Detection / classification | YOLO11 → int8 TFLite |
| Monocular depth | Depth Anything V2 → ExecuTorch (`.pte`) |
| On-device runtimes | `react-native-fast-tflite`, `react-native-executorch` |
| App | Expo / React Native (TypeScript), development build — not Expo Go |
| Training, export, reference implementation | Python (`src/`) |
| Experiment tracking | MLflow |
| Dataset versioning | Roboflow |

Full design detail: [`docs/architecture.md`](docs/architecture.md). The numbers this
design is held to: [`docs/requirements.md`](docs/requirements.md). Contributing:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Current status

**This repository is the Lesson 01 scaffold. Almost nothing here is implemented yet.**

- `src/smart_scene_analyzer/` contains module docstrings, dataclass field declarations,
  and documented planned function signatures — **no function bodies**. Nothing in `src/`
  runs inference, trains a model, or exports an artifact today.
- `app/` is an **empty directory** containing only a README explaining why. Lesson 05
  is what creates the Expo app. There is no mobile app to build or run yet.
- `tests/` holds one test file, `tests/test_units_guard.py`, which exercises the
  `units_guard` hook rather than any project code (there is no project code to test yet).
- `docs/architecture.md` and `docs/requirements.md` are design documents, not records of
  a built system — read the "open decisions" and "not yet resolved" sections in each
  before assuming something is settled.
- No model has been trained, no dataset version has been generated, and no artifact has
  been exported. Any number that looks like a metric in this repo's docs (e.g. the N3
  mAP figure) is a **target**, not a measured result — none currently exist.

If you are looking for something to run, `uv sync` and `uv run pytest` are the whole of
it today.

## Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.11 — pinned in `.python-version`, floor declared in `pyproject.toml` | Everything under `src/` and `tests/` |
| [`uv`](https://docs.astral.sh/uv/) | Recent release | Python dependency management. Never `pip install` into the system interpreter |
| Git | Any recent version | Cloning this repo |

The following are **not needed yet** — they apply from Lesson 05, when `app/` is
created:

| Tool | Needed for |
|---|---|
| Node.js / npm | The Expo app |
| Xcode (macOS only) | The iOS Simulator |
| Android Studio | The Android Emulator |

## Setup

From a fresh clone:

```bash
git clone <this-repo-url> smart-scene-analyzer
cd smart-scene-analyzer
uv sync
cp .env.example .env
```

You do not need to install Python yourself. `.python-version` pins the project to 3.11,
so `uv sync` downloads and uses that interpreter regardless of what your system Python
is. The pin is deliberate rather than cosmetic: `mypy` is configured to check against
3.11, and on a newer interpreter it fails inside NumPy's bundled type stubs before it
reaches any of this project's code.

`uv sync` installs **only** the base dependencies plus the `dev` dependency group
(`pytest`, `ruff`, `mypy`, and friends) — see `pyproject.toml`. It does **not** install
the dataset, ML, or depth packages; those are optional extras, added in the lesson that
first needs them, so nothing pulls in `torch` or `ultralytics` before there is a reason
to have them installed:

```bash
uv sync --extra dataset   # roboflow, h5py, scipy      (Lesson 02)
uv sync --extra ml        # ultralytics, torch, mlflow (Lesson 03)
uv sync --extra depth     # transformers, timm         (Lesson 03)

# or combine what you need in one call:
uv sync --extra dataset --extra ml --extra depth
```

After `cp .env.example .env`, open `.env` and fill in `ROBOFLOW_API_KEY` and
`ROBOFLOW_WORKSPACE` if and when a lesson needs them. `.env` is gitignored; never commit
it. See `.env.example` for what each variable is for and why none of them is a runtime
secret.

## Running it

**There is nothing to run beyond the Python tooling right now.** `src/` is a reference
implementation with no function bodies yet, and `app/` — the actual product a user would
install — does not exist until Lesson 05. There is no server to start and no `python -m`
entry point that produces a result today.

Once Lesson 05 creates `app/`, running the app will require a development build (both
ML runtimes are native modules, so it cannot run inside Expo Go):

```bash
cd app && npm install
npx expo run:ios        # build + install to the iOS Simulator
npx expo run:android    # build + install to the Android Emulator
```

Those commands do not work yet, because `app/` does not exist. See `app/README.md` for
the fuller explanation of why the app is deliberately left out of this scaffold.

## Running tests, lint, and type checks

```bash
uv run pytest                          # run the test suite
uv run pytest -m "not integration"     # skip anything needing model weights or exported artifacts on disk
uv run ruff check .                    # lint
uv run ruff format .                   # format
uv run mypy src                        # type check src/ (see pyproject.toml: strict = true)
```

The test suite is designed to run with no GPU, no network access, and no model weights
on disk (`.claude/skills/offline-suite`). Tests that need a checkpoint or an exported
artifact carry the `integration` marker and are excluded by
`uv run pytest -m "not integration"`. Today, only `tests/test_units_guard.py` exists, and
it needs none of those things.

Once `app/` exists (Lesson 05), the app has its own type check:

```bash
cd app && npx tsc --noEmit
```

## Project layout

| Path | Contents |
|---|---|
| `src/smart_scene_analyzer/` | Python: training, export, and the reference implementation. Currently docstrings and type declarations only |
| `app/` | The Expo app (TypeScript). **Empty until Lesson 05** — see `app/README.md` |
| `app/assets/models/` | Where the bundled `.tflite` and `.pte` artifacts will live once exported. Gitignored — what is there is what runs |
| `app/src/fusion/` | The planned TypeScript port of `src/smart_scene_analyzer/fusion.py`. Must agree with it on shared fixtures |
| `tests/` | Unit, integration, and regression tests, including export parity. Currently one file, testing the units-guard hook |
| `docs/` | Architecture, requirements, decisions, and the two budgets |
| `docs/architecture.md` | The system design: components, data contracts, latency budget, open decisions |
| `docs/requirements.md` | Functional and non-functional requirements, with targets — not measured results |
| `docs/decisions/` | ADRs — one file per architectural decision |
| `docs/credit-budget.md` | The Roboflow credit ledger. Read before any billed call |
| `docs/artifact-budget.md` | Model and app size budget — the one that fails at install time, not billing time |
| `data/` | Dataset exports. Gitignored. Versioned in Roboflow, not git |
| `notebooks/` | Exploration only. Nothing here is production code |
| `.claude/agents/` | Subagent definitions — the engineering roles |
| `.claude/skills/` | This project's own procedures |
| `.claude/hooks/` | Enforcement scripts that run on tool calls, two of which block |

## Further reading

- [`docs/architecture.md`](docs/architecture.md) — the system design and the contracts
  between its components
- [`docs/requirements.md`](docs/requirements.md) — what this system is required to do,
  and the targets it is held to
- [`docs/decisions/`](docs/decisions/) — ADRs, including the decision to run both models
  on-device rather than serving them
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development environment, conventions, and gates
- [`CLAUDE.md`](CLAUDE.md) — the full set of project rules loaded into every Claude Code
  session in this repo
- [`app/README.md`](app/README.md) — why `app/` is empty and what lands there in
  Lesson 05
