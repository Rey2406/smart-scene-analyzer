# Contributing

This project is a course scaffold, built up lesson by lesson. Most of what is described
below is process that applies as soon as there is code to change — read
[`README.md`](README.md)'s "Current status" section first if you have not, so you know
what actually exists right now.

## Development environment

```bash
git clone <this-repo-url> smart-scene-analyzer
cd smart-scene-analyzer
uv sync
cp .env.example .env
```

`.python-version` pins the project to Python 3.11, so `uv sync` provisions that
interpreter for you — do not build the environment on a newer one. `mypy` is configured
to check against 3.11, and on a newer interpreter it fails inside NumPy's bundled type
stubs before it reaches any project code.

`uv sync` installs the base dependencies plus the `dev` group (`pytest`, `ruff`, `mypy`).
It does not install the dataset (`roboflow`, `h5py`, `scipy`), ML (`ultralytics`,
`torch`, `mlflow`), or depth (`transformers`, `timm`) packages — those are optional
extras in `pyproject.toml`, added with `--extra`:

```bash
uv sync --extra dataset --extra ml --extra depth
```

Add only the extras the work you're doing needs. Never `pip install` into the system
interpreter — `uv` owns the environment.

Fill in `.env` from `.env.example` if your work touches Roboflow or MLflow.
`ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, and `MLFLOW_TRACKING_URI` are dev-time only;
none of them is a runtime secret, because the shipped app has no runtime service to
authenticate to. `.env` is gitignored — never commit it.

For work on `app/` (from Lesson 05 onward, once it exists): `npm install` inside `app/`.
`uv` never touches `app/`, and `npm` never touches the Python side.

## Branch naming and commits

- Branch names: `<type>/<short-description>`, e.g. `feat/letterbox-transform`,
  `fix/nms-tiebreak`, `docs/architecture-latency-budget`.
- Commit messages: a short imperative summary line, optionally followed by a body
  explaining *why* — not just what changed. If a commit touches a contract in
  `docs/architecture.md` §4 or a value in `docs/requirements.md`, say so in the body.
- Keep commits scoped to one concern. A commit that touches both `src/` and `app/`'s
  fusion port should say, in the message, which side was found wrong and which was
  brought into agreement — see "Two boundaries" below for why that distinction matters.

## Gates a change must pass before review

Run these locally before opening a PR — CI runs the same commands (N18, per
`docs/requirements.md`):

```bash
uv run ruff format .                   # format
uv run ruff check .                    # lint
uv run mypy src                        # type check src/ (strict = true in pyproject.toml)
uv run pytest                          # full test suite
uv run pytest -m "not integration"     # the subset that must run with no GPU, no network, no weights on disk
```

Notes on what each gate actually checks:

- **`ruff format` / `ruff check`** — formatting and lint rules configured in
  `pyproject.toml` (`[tool.ruff]`). `format_python.py` runs these automatically after
  `Write`/`Edit` inside a Claude Code session, but that does not substitute for running
  them yourself before pushing.
- **`mypy src`** — `[tool.mypy]` in `pyproject.toml` sets `strict = true` against Python
  3.11. Every public function needs type hints and a docstring stating units and shapes
  (a depth value that does not say whether it's metric or relative is a real source of
  bugs in this project).
- **`pytest`** — every module in `src/` needs a matching test file. Model inference is
  mocked in unit tests; nothing in the default suite may require a GPU, network access,
  or weights on disk. A test that genuinely needs a checkpoint or an exported artifact
  carries `@pytest.mark.integration` so `pytest -m "not integration"` still runs clean —
  see `.claude/skills/offline-suite`.
- **Export parity and fusion port parity are tests, not manual checks.** N20 (exported
  artifact vs. PyTorch reference) and N21 (`app/src/fusion/` vs.
  `src/smart_scene_analyzer/fusion.py`) run against the *same* fixture set on both sides.
  If you touch either implementation, run the shared fixtures against both, and if they
  disagree, that disagreement is the finding — fix the one that's wrong, or record why
  the reference is wrong. Never edit the fixtures to make them agree.

For `app/` (once it exists): `cd app && npx tsc --noEmit` is the equivalent type-check
gate, per N17.

## Working with the `.claude/agents/` roles

Work in this project is scoped to specialized subagents, each owning one slice of the
system:

| Role | Owns |
|---|---|
| Architecture | System design, folder structure, engineering plan |
| Documentation | README, architecture docs, contribution guide, model cards |
| DevOps | Dev environment, mobile builds, GitHub Actions, releases |
| Dataset Engineer | Roboflow projects, annotation review, versioning, quality analysis |
| Data Pipeline | Preprocessing, augmentation, conversion, validation scripts |
| ML Engineer | Training, fine-tuning, hyperparameters, MLflow, run diagnosis |
| Evaluation | Metrics, confusion matrices, run comparison, model cards |
| Export | Export, quantization, numerical parity against the reference |
| QA | Unit, integration, and regression tests |
| Integration | Fusing detections with depth into one scene result |
| Mobile | The Expo app: on-device inference, fusion, overlays, capture, latency |

Read the role's own file in `.claude/agents/` before invoking it — its constraints are
part of the work, not boilerplate. Two boundaries between roles are **load-bearing**,
not just tidy, and should not be worked around even if it would be more convenient in
the moment:

- **Evaluation has no `Edit` tool.** It reports on models; it cannot change the code
  that produces them. If Evaluation could adjust a model it disliked the number for, it
  would no longer be measuring anything.
- **Mobile does not change the exported artifacts, and Export does not change the app.**
  When the on-device result disagrees with the Python reference, that disagreement
  *is* the finding. If one role could edit both sides, it would resolve the disagreement
  by adjusting whichever side was easier to reach — and nobody would ever learn which
  side was actually wrong.

If you find yourself wanting one role to reach across either of these boundaries, that
is a sign to stop and hand the finding to the other role, not to widen the role's tools.

## `.claude/hooks/` that block

Two hooks in `.claude/hooks/` return a hard block (exit code 2) rather than merely
warning:

| Hook | Fires on | Blocks when |
|---|---|---|
| `credit_gate.py` | Before billed Roboflow MCP tool calls | The ledger (`docs/credit-budget.md`) has no pending row — a numeric **Estimated** with a blank **Actual** — that fits the remaining balance, or the call would breach the 0.5-credit reserve floor |
| `units_guard.py` | Before `Write`/`Edit` to `src/` **or** `app/`, in `.py`, `.ts`, or `.tsx` | The text contains a metric-depth identifier or claim (`depth_meters`, "2.3 m away", etc.) |

**If a hook blocks you, the correct response is to satisfy it — never to edit the hook,
and never to route around it by writing the file some other way.** For `credit_gate.py`,
that means recording a pending estimate in the ledger before the call. For
`units_guard.py`, that means renaming the field to the sanctioned vocabulary
(`relative_depth`, `relative_inverse_depth`, `depth_score`) or rewording the prose.

If you believe a hook is genuinely wrong — not merely inconvenient at this moment — say
so and stop. That is an ADR (`.claude/commands/adr.md` writes one into
`docs/decisions/`), not a patch to the hook.

## What must never be committed

- **Secrets.** `.env`, API keys, `*.pem`, `*.key`. `.env.example` documents which
  variables are required by name; it never contains a value that looks real.
- **Datasets.** `data/` is gitignored. Dataset changes go through Roboflow, so the
  change is versioned and reviewable — do not modify `data/` directly.
- **Model weights and exported artifacts.** `*.pt`, `*.onnx`, `*.mlpackage`, `*.tflite`,
  `*.pte`, `*.engine` are all gitignored, including inside `app/assets/models/`. They
  are produced by the export step and tracked through MLflow's registry versions, not
  through git. If an artifact is missing after a fresh clone, that is expected —
  re-export it rather than searching for it in version control.
- **Generated native app directories.** `app/ios/` and `app/android/` (once `app/`
  exists) are written by `expo prebuild` and gitignored; the next prebuild discards any
  hand-edit made there. Native configuration belongs in `app.json` or a config plugin.
- **`.claude/settings.local.json`.** Per-session permission accretion, not reviewed
  policy — `.claude/settings.json` is the committed, reviewed version.

## Review expectations

- A PR that touches a data contract (`docs/architecture.md` §4, C1–C13) states which
  contract changed and why, and updates the contract's documentation in the same PR.
- A PR that changes a value `docs/requirements.md` marks as a target (an mAP figure, a
  latency allocation) does not also claim it as a measured result unless it comes from
  MLflow or a committed evaluation report.
- A PR touching `app/src/fusion/` or `src/smart_scene_analyzer/fusion.py` includes the
  shared-fixture run showing both sides agree, or states explicitly why they don't yet
  and which side is being tracked as the finding.
- Do not commit or push on someone else's behalf without being asked.
