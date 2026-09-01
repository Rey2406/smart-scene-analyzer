# Roadmap — Weeks 2 to 6

- **Status:** plan, Lesson 01. Nothing in it has been executed.
- **Date:** 2026-09-01
- **Held to:** [`docs/requirements.md`](requirements.md), [`docs/architecture.md`](architecture.md)
- **Constrained by:** [ADR 0001 — Inference target: on-device](decisions/0001-inference-target.md),
  [`docs/credit-budget.md`](credit-budget.md), [`docs/artifact-budget.md`](artifact-budget.md)

Five phases, in dependency order. **No dates and no durations appear in this document.**
Order and entry/exit conditions are the plan; a calendar is a separate artifact and a
worse one, because a date slipping tells you nothing about what is now unsafe to start.

Every exit criterion below is written so that **someone who was not in the room can check
it** from the repository, the MLflow registry, the Roboflow platform, or a device. "Model
trained" is not an exit criterion. "Detector V1 registered with mAP@50 recorded against
the held-out test split of dataset version 1" is.

Four decisions are open, and this plan does not close any of them. Where a phase depends
on one, the dependency is stated and the phase is marked. See
[Blocked on decisions](#blocked-on-decisions) for the latest point at which each must be
resolved.

---

## Phase map

| Phase | Produces | Consumed by | Credits |
|---|---|---|---|
| **W2 — Dataset Engineering** | Taxonomy, dataset version 1, split validation, the N4 pair set | W3, W4 (label order) | **4.0** allocated |
| **W3 — Model Development** | Detector V1 in the registry, depth reference run, model cards | W4 | **3.0** allocated |
| **W4 — Export, quantization, parity** | `.tflite`, `.pte`, `model_contract.json`, C12 fixtures, N20 | W5 | **0.0** — see the flag below |
| **W5 — On-device inference and delivery** | The Expo development build, measured N1/N6/N8/N9, device-vs-reference parity | W6 | **0.0** structurally |
| **W6 — CI/CD/CT** | `ci.yml`, `ct.yml`, promotion policy, one CT run | — | **4.0** allocated |

> **⚠️ Credit-ledger flag.** `docs/credit-budget.md` allocates **2.0 credits to
> "Lesson 04 — Backend Engineering"** for "one hosted-API latency and cost comparison."
> ADR 0001 deleted the backend. W4 as scoped here is export and parity, which touches no
> Roboflow endpoint and spends nothing. Those 2.0 credits are either **returned to the
> buffer (7.0 → 9.0)** or spent on something that has not been named. Somebody has to
> decide; this document does not, because the ledger is the source of truth and revising
> it "silently" is exactly what it forbids.

---

## W2 — Dataset Engineering

**Owning role:** `dataset-engineer` (Roboflow platform: taxonomy, upload, versioning) with
`data-pipeline` (local validation code). Review of the taxonomy document by `documentation`.
All three role files exist.

### Deliverables

| Artifact | Exists or does not |
|---|---|
| `docs/taxonomy.md` | The ordered class list of length K, with an inclusion rule per class. **The order in this file is the order in C11's `labels` array** — it is an index, not a list |
| `docs/decisions/0002-object-taxonomy.md` | ADR, status `accepted`, naming the K classes and what was excluded |
| Roboflow dataset **version 1** | Immutable, referenced everywhere by its version number, never as "the latest" |
| `docs/dataset-card.md` | Version number, source, per-class image and instance counts for train/val/test, split policy, and **the version's preprocessing and augmentation settings copied out verbatim** |
| `src/smart_scene_analyzer/training/dataset.py` | Implemented: dataset version → training inputs. Docstring names the version number |
| `tests/data/test_dataset_integrity.py` | Label-index range, box degeneracy, split disjointness by image hash |
| The **N4 ordering pair set** | Annotated object pairs on the test split, in a format `evaluation/depth_ordering.py` can load. **Its storage location is an open decision** — see [Blocked on decisions](#blocked-on-decisions) |
| `docs/credit-budget.md` | Ledger rows with estimates *and* actuals, `Remaining` updated, reconciled against the platform usage page on a stated date |
| `docs/requirements.md` N3 | The provisional marker replaced with a target stated over K classes, or an explicit note that it stays provisional and why |

### Entry conditions

1. ADR 0001 is `accepted`. (It is.)
2. `docs/credit-budget.md` holds a starting balance and a W2 allocation. (It does: 4.0.)
3. `ROBOFLOW_API_KEY` and `ROBOFLOW_WORKSPACE` resolve from the environment, and the
   `Workspace` row of the ledger names a real slug — it currently says
   `<your-workspace-slug>`.
4. A human is available to accept the taxonomy ADR. This is a product decision; the
   dataset-engineer proposes, it does not decide.
5. The `credit-ledger` and `dataset-qa-sweep` skills have been read.

### Exit criteria

- [ ] `docs/taxonomy.md` exists and lists exactly K classes in a **numbered** order, and
      `docs/decisions/0002-object-taxonomy.md` has status `accepted` and the same list.
- [ ] A Roboflow dataset version **number** appears in `docs/dataset-card.md` and in the
      docstring of `training/dataset.py`. Searching the repo for the string "latest" as a
      dataset reference returns nothing.
- [ ] `docs/dataset-card.md` records per-class image and instance counts for all three
      splits, and **no class has zero instances in the test split** — a class with no test
      instances cannot appear in N3 and its absence will not be noticed.
- [ ] `uv run pytest tests/data/test_dataset_integrity.py` passes against the exported
      version: every label index `< K`, every box satisfies `0 ≤ x1 < x2 ≤ 1` and
      `0 ≤ y1 < y2 ≤ 1`, and no image hash appears in two splits.
- [ ] The version's preprocessing block is copied verbatim into `docs/dataset-card.md`,
      and a line states whether it agrees with architecture A2/A4 (`DET_SIZE = 640`,
      aspect-preserving letterbox, pad `(114,114,114)`) or deliberately differs.
- [ ] The N4 pair set exists at a path recorded in `docs/dataset-card.md`, holds a stated
      number of annotated pairs drawn from the **test** split, and
      `evaluation/depth_ordering.py`'s loader has a passing test against it — **or** an
      accepted ADR records that N4 is measured some other way.
- [ ] `docs/credit-budget.md`: phase spend ≤ 4.0, `Remaining` updated, `Last reconciled`
      carries a real date and a real workspace.

### Dependencies

None on later phases. This is the root of the graph. Everything downstream inherits the
label order fixed here.

### Risk most likely to derail this phase

**Train/serve preprocessing skew, introduced by the Roboflow version's resize setting.**

Roboflow's default version preprocessing stretches to a square. Architecture A2/A4 says the
app letterboxes with aspect preservation and `(114,114,114)` padding. If version 1 is
generated with a stretch resize, the detector trains on stretched objects and W5 feeds it
letterboxed ones with grey bands. Both run. mAP in W3's report is measured on the stretched
distribution and is *right*; mAP on the phone is worse and nobody can localize it — and the
`src/` reference will agree with the app rather than with training, so W4's N20 parity test
passes cleanly while the whole pipeline is wrong. The failure surfaces in W5, four phases
from its cause, on the one boundary that costs an app-store release to fix.

**Early warning sign:** the version is generated and `docs/dataset-card.md` does not yet
contain the preprocessing block. If nobody wrote it down at generation time, nobody
compared it to A2/A4.

**Secondary:** the taxonomy ADR arriving *after* version 1 is generated. A regenerated
version bills again, invalidates every count in the dataset card, and resets N3's baseline.
Tripwire: any `versions_create` call while `docs/decisions/` has no accepted taxonomy ADR.

---

## W3 — Model Development

**Owning roles:** `ml-engineer` (training, MLflow, the depth reference run) and
`evaluation` (measurement, model cards). Both files exist. **The separation is
load-bearing here:** `evaluation` has no `Edit` tool, so it cannot adjust the model it is
grading. Do not route a "the number is disappointing" follow-up through it.

### Deliverables

| Artifact | Exists or does not |
|---|---|
| `src/.../training/train_detector.py` + a committed run config | Runnable entry point; every hyperparameter in the config, none in the code |
| `src/.../training/tracking.py` | `RunTracker` implementation + `NullTracker`. **No module names a backend** (architecture §8.4, C13) |
| MLflow experiment, ≥ 1 run | Seed, dataset version number, full config, metrics (N19) |
| **Detector V1**, registered | An explicit registry version number, referenced by number in code and docs |
| `src/.../inference/depth.py` | The Depth Anything V2 reference backend. **Run, not trained** — depth is inference-only in this project |
| `src/.../evaluation/detection_metrics.py`, `depth_ordering.py` | Implemented, with unit tests over synthetic arrays that need no weights |
| `docs/model-cards/detector-v1.md` | mAP@50 overall and **per class**, on the held-out test split of dataset version 1 |
| `docs/model-cards/depth-v1.md` | Checkpoint identifier, `DEPTH_SIZE`, the output semantics string, N4 pairwise ordering accuracy and the pair count |
| Workstation baseline timings | fp32 PyTorch inference time for both models, recorded. This is W4's tolerance input and W5's sanity check |
| `docs/decisions/000X-mlflow-hosting.md` | **If the decision is taken here.** See [Blocked on decisions](#blocked-on-decisions) |

### Entry conditions

1. All W2 exit criteria met. K is known and frozen.
2. The N4 pair set exists and loads, **or** an accepted ADR has re-scoped N4. Without one
   of these, `docs/model-cards/depth-v1.md` cannot be completed and W4 has no way to detect
   that quantization broke depth ordering.
3. `MLFLOW_TRACKING_URI` resolves to *something*. The `RunTracker` protocol means this need
   not be the final hosting choice — but see the note in Blocked on decisions about
   migrating runs.
4. `uv run pytest -m "not integration"` is green on `geometry/` and `preprocess/`, since the
   training data path uses the same letterbox arithmetic the app will.

### Exit criteria

- [ ] The MLflow registry contains a model named `detector` at **version 1**, and its
      source run record contains: the seed, the dataset **version number**, the complete
      config, and `map50` measured on the **test** split (not val).
- [ ] `docs/model-cards/detector-v1.md` states mAP@50 overall and per class, names the
      dataset version, and either meets N3's target or records the gap with a diagnosis
      classified by the `error-triage` skill as DATA, TAXONOMY, or MODEL.
- [ ] `docs/model-cards/depth-v1.md` records N4 pairwise ordering accuracy with the number
      of annotated pairs stated. A card whose N4 row says "TBD" does not satisfy this.
- [ ] Reproducibility check: a second person can read the run record and reconstruct the
      command. Concretely — the run record contains no field whose value is only in
      somebody's shell history.
- [ ] `uv run pytest -m integration -k reference_pipeline` passes: one image through
      `src/`, end to end, producing a `SceneResult` with at least one non-null
      `relative_depth`. **The Python reference works before anything is exported.**
- [ ] `grep -rn "mlflow" src/` returns hits only in `training/tracking.py`.
- [ ] `docs/credit-budget.md`: phase spend ≤ 3.0, reconciled. Local `ultralytics` training
      costs nothing; any hosted run has an estimate row *and* an actual row.

### Dependencies

W2, entirely. The label order, the split definition, and the N4 pair set all come from it.

### Risk most likely to derail this phase

**N4 has no ground truth and no storage path, so depth quality is unmeasured — and stays
unmeasured through W4's quantization.**

Depth Anything V2 is inference-only here, so there is no training loop that would force the
question. A Roboflow bounding-box dataset has no depth channel and no pairwise-ordering
annotation type. `data/` is gitignored and assistants must not modify it. So the pair set
N4 requires has, right now, nowhere to live that is versioned and reviewable. The tempting
resolution is to implement `depth_ordering.py` against synthetic arrays, ship green unit
tests, and write "TBD" in the model card — at which point W4 quantizes a ViT with **no
baseline to regress against**, and losing depth ordering to int8 is precisely the failure
mode `.claude/agents/ml-engineer.md` warns is invisible to an error metric.

**Early warning sign:** `tests/` contains passing tests for `depth_ordering.py` and not one
of them loads a real annotated pair. Green tests over synthetic data are the shape this risk
takes when it is still cheap to fix.

**Secondary:** hosted training over-running its credit estimate. GPU training bills at 1
credit per 30 minutes and the phase has 3.0. Tripwire: an estimate row in the ledger whose
epoch count implies more than 90 minutes of wall time.

---

## W4 — Export, quantization and parity

**Owning roles:** `ml-engineer` wearing the **Export** hat, `integration` (the fusion
reference and the C12 fixture set), `qa` (the parity tests and the offline suite).

> **Role gap.** `CLAUDE.md`'s role table lists **Export** as a distinct role pointing at
> `.claude/agents/ml-engineer.md`. There is no `export.md`. In practice the boundary that
> matters — "Export does not change the app" — is already written into `ml-engineer.md`, so
> nothing is unenforced. But the table promises a file that does not exist. Either create
> `.claude/agents/export.md` or correct the table; do not leave a reader hunting for it.

### Deliverables

| Artifact | Exists or does not |
|---|---|
| `src/.../fusion.py` | The reference Scene Understanding Layer (architecture §5). Pure |
| `src/.../scene.py` | `SceneResult` assembly, including the N12 depth-failed path |
| `tests/fixtures/fusion/*.json` | The **C12 fixture set**. Plain JSON. One fixture per row of architecture §5.2 |
| `app/assets/models/*.tflite` | YOLO11 int8. Gitignored |
| `app/assets/models/*.pte` | Depth Anything V2. Gitignored |
| `app/assets/models/model_contract.json` | **C11. Committed** — it is the only part of the artifact contract a reviewer can read |
| `src/.../export/parity.py` + its test | N20, with the tolerance written in the test, not in a comment |
| `docs/model-cards/detector-v1-int8.md` | **Per-class** mAP delta fp32 → int8, and the calibration set named |
| `docs/artifact-budget.md` | Detection model fp32 and int8, depth model, combined total — each with a date. **Targets filled in before the first measurement** |
| **The concurrency probe result** | Both exported artifacts run on two threads on the workstation; wall time recorded as `max(a,b)` or `a+b`. See the W5 risk |
| `docs/decisions/000X-one-runtime-or-two.md` | If the size measurements force §10.4 |

### Entry conditions

1. Detector V1 is in the registry and its checkpoint is retrievable **from the registry**,
   not from somebody's `runs/` directory.
2. The taxonomy is frozen. Label order is C11's highest-risk field and the export bakes it in.
3. `docs/artifact-budget.md`'s **Target** column is filled in. The file says it plainly: a
   target chosen after seeing the number is a description, not a budget. Fill it first.
4. The depth checkpoint is pinned to an exact identifier.
5. C12's schema (architecture §4) is agreed with whoever will write the TypeScript suite in
   W5 — the fixture format is a two-language contract and changing it later invalidates both.

### Exit criteria

- [ ] `uv run pytest -m integration -k parity` passes: the `.tflite` output and the PyTorch
      reference agree on the shared fixtures within a tolerance **stated in the test**, and
      likewise the `.pte`.
- [ ] `uv run pytest -m "not integration"` passes with **no GPU, no network, and no weights
      on disk**, at ≥ 80% coverage on `src/` (N16), and `uv run mypy --strict src` is clean
      (N17).
- [ ] `app/assets/models/model_contract.json` is committed and contains, for the detector:
      file, sha256, registry version, input shape/dtype/layout/quant params, **one entry per
      output tensor** with index, shape, dtype, `encoding` (`cxcywh` or `xyxy`) and
      `normalized`; and an ordered `labels` array of length K that matches
      `docs/taxonomy.md` **line for line**. A diff of the two orders is empty.
- [ ] The depth block of C11 carries
      `semantics: "relative-inverse-depth-larger-is-nearer"` as a literal string.
- [ ] `docs/artifact-budget.md` has a measured size and a date for the detection model, the
      depth model, and the combined total, and the total is **≤ 40 MB (N9a)** — or an
      accepted ADR records the overage and names what gives.
- [ ] The C12 fixture set covers all seven degenerate cases in architecture §5.2, and every
      fixture loads with a plain `json.load` — no `.npy`, no pickles, no NaN literals.
- [ ] The fusion tests assert against the fixtures' `expected` blocks. **No test regenerates
      an expectation.** When the two disagree, the disagreement is the finding.
- [ ] `docs/model-cards/detector-v1-int8.md` reports the per-class mAP delta. An aggregate
      that barely moved while one class collapsed is the documented failure mode; the
      per-class table is what catches it.
- [ ] The concurrency probe result is recorded, with the thread count and the host CPU.
- [ ] **Credit spend: 0.0.** Nothing in this phase touches a Roboflow endpoint.

### Dependencies

- W3 for the checkpoint and its registry version.
- W2 for the label order baked into C11.
- **`fusion.py` and the C12 fixtures depend on neither.** See the note below.

> **Note on sequencing.** The fixed phase list places "model fusion" in W4. Architecture
> §8.2 makes fusion pure by construction — arrays in, a dataclass out, no model, no clock,
> no files — and its tests run against a hand-written 8×8 depth grid. It therefore has **no
> dependency on W2 or W3 at all** and could be built and fixture-covered in parallel with
> either. Keeping it in W4 is a scheduling choice, not a technical constraint, and it puts
> the C12 fixture set — which W5's port parity depends on — on the critical path behind two
> phases that cannot influence it. Consider starting it early; the plan does not require it.

### Risk most likely to derail this phase

**Depth Anything V2 does not export to `.pte` within N9a's 40 MB combined budget, forcing
the §10.4 one-runtime-or-two decision mid-phase.**

The ViT is, per `docs/artifact-budget.md`, "the harder one to shrink," and nobody has
measured it. If the `.pte` alone approaches or exceeds 40 MB, the resolution is not a
tuning pass — the artifact budget calls consolidation "a real workstream," and both
directions (YOLO11 → ExecuTorch, or Depth Anything V2 → TFLite) are unverified. Executing
one inside the phase that discovered the need for it is how W4 absorbs W5.

**Early warning sign:** the size of the **fp32** `.pte` at the *first successful export*,
before any quantization. Record it then, not after. If fp32 is more than roughly 4× the
budget line, quantization will not recover it and the §10.4 decision is already due.

**Secondary:** the exported detector's output encoding is not what the app will assume.
Ultralytics emits `cxcywh` normalized to `[0,1]` on some export paths and `DET_INPUT` pixels
on others. Both decode without error; one of them draws boxes in the wrong place at the
wrong scale. Tripwire: C11's `outputs[].encoding` field filled in from memory rather than
from an inspection of the exported graph.

---

## W5 — On-device inference and mobile delivery

**Owning roles:** `mobile` (the app, the port, the overlays, on-device latency),
`devops` (the development build, native config, bundling, size accounting), `qa`
(device-vs-reference parity). All three exist.

> **Boundary reminder.** Mobile does not re-export an artifact and Export does not edit
> `app/`. When the device disagrees with the reference, **the disagreement is the finding**,
> and a single role owning both sides would resolve it by adjusting whichever was easier to
> reach — destroying the only information that mattered, which is *which side is wrong*.

### Deliverables

| Artifact | Exists or does not |
|---|---|
| An Expo **development build** that installs on iOS and Android | Not Expo Go — both runtimes are native modules |
| `app/app.json` | Both runtimes' config plugins, camera and library permissions |
| `app/metro.config.js` | `assetExts` includes `tflite` and `pte`, or the bundler silently omits the models |
| `app/src/fusion/index.ts` | The **port** of `src/.../fusion.py` |
| `app/src/fusion/__tests__/` | Runs the **same** `tests/fixtures/fusion/*.json` files. Not copies |
| `app/src/models/ModelHost.ts` | Loads and warms **both** models at mount (R9), exposing loading / ready / failed |
| `app/src/geometry/` | `letterboxInverse` and `toScreen` — the only named conversions, each with both spaces in its signature |
| The overlay | Boxes, labels, depth shading, and three visually distinct states (F8) |
| `docs/latency-report.md` | p50 and p95 over a stated run count, **broken down by stage S0–S8**, on both reference devices, each stage compared to its §6.1 allocation |
| `docs/artifact-budget.md` | Delivered install size per platform **with the method named**, peak RSS with both models loaded, cold model load time |

### Entry conditions

1. W4's exit criteria met: both artifacts exist, N20 passes, `model_contract.json` is committed.
2. The C12 fixture set is **frozen**. A fixture edited during W5 stops being a two-language
   contract and becomes a way to make the port agree.
3. The `one runtime or two` decision is resolved, or W4's size measurements have confirmed
   two runtimes fit. Starting the app against two runtimes and consolidating mid-phase means
   rebuilding the runtime layer.
4. A **device floor candidate** is named, so the right handsets are in the room. The floor
   is *confirmed* by this phase's RSS measurement, but a candidate is needed to plan the
   measurement — see Blocked on decisions.
5. Physical or emulated iPhone 13 and Pixel 6a available. N1 is meaningless without its
   named device.

### Exit criteria

- [ ] `cd app && npx expo run:ios` and `npx expo run:android` each produce a build that
      installs, and analyzing a 1280×720 image draws boxes with depth shading on both.
- [ ] The TypeScript fusion suite passes on the **identical fixture files** the Python suite
      uses, within C12's tolerance. A test asserts the fixture path resolves into
      `tests/fixtures/fusion/` rather than into `app/`.
- [ ] **Airplane mode:** radio off, full analysis, result identical to the same image with
      the radio on (F10, N1a). A packet capture during one analysis shows zero bytes.
- [ ] `docs/latency-report.md` exists with p50 and p95 per stage over a stated run count on
      both reference devices. **N1 either passes or the report names the stage that exceeded
      its allocation.** A miss is a recorded finding about that stage, not a phase failure —
      architecture §6 says the allocations are ceilings to be measured against, not a table
      to re-cut when the measurement is inconvenient.
- [ ] N6 measured: cold load of both models, timed from mount, ≤ 3.0 s or the overage
      recorded per component against §7's allocation.
- [ ] `docs/artifact-budget.md` records delivered install size for both platforms **with the
      measurement method stated** (`bundletool build-apks --connected-device`, or store
      reporting), and peak RSS with both models loaded and one analysis in flight against
      N8's 1.2 GB.
- [ ] `cd app && npx tsc --noEmit` is clean (N17).
- [ ] Three states demonstrated, as three passing tests or three screenshots: a
      zero-detection image renders the **empty-success** state; a deliberately corrupted
      `.pte` renders the **detection-only degraded** state (N12) with `depth_available =
      false` and no crash; a missing `.tflite` renders the **failed** state. All three are
      visually distinct from each other and from loading (F8).
- [ ] **Device-vs-reference parity:** the same input image through `src/` and through the
      app produces the same detections and the same `relative_depth` ordering, within a
      stated tolerance. Where they differ, the difference is written down, not resolved by
      editing whichever side was closer to hand.
- [ ] Credit spend **0.0, structurally** — the app ships with no credentials, so no Roboflow
      call is possible from it.

### Dependencies

- W4 for both artifacts, C11, and the frozen fixture set.
- W2 for the label order, which C11 carries and the overlay renders. An app built against
  last week's order runs, draws boxes, and names them wrong, and no test on either side of
  the boundary catches it.

### Risk most likely to derail this phase

**The two runtimes cannot execute concurrently, and N1 is not achievable as budgeted.**

Architecture A12 — that the two runtimes execute on separate native threads — is
**unverified**, and §6.2 shows exactly what rests on it. The serial arrangement is
`35 + 20 + 25 + 90 + 150 + 20 + 15 + 30 + 15 = 400 ms`: the whole of N1 with **zero
reserve**. A p95 target met with no margin is not met; the first GC pause that lands inside
it fails. The 55 ms reserve in §6.1 *is* the `240 − 185` that concurrency buys, and nothing
else in the budget produces it.

So concurrency is **a requirement of this design, not an optimization.** If the JS↔native
bridge serializes the two calls, or the runtimes contend on one core cluster, or §10.4
consolidated them onto a single runtime with a single inference queue, then one of three
things must give — `DEPTH_SIZE` drops from 518 to 256 (trading N4 accuracy), the detector
drops to a smaller YOLO11 variant (trading N3), or N1 moves. **All three are export or
requirements changes, and none of them is a W5 change.** Discovering this on device means
reopening W4 from inside W5.

**Early warning sign — and it appears in W4, not here:**

1. **The concurrency probe, run in W4** (listed among W4's deliverables): execute both
   exported artifacts on two threads on the workstation and record whether wall time is
   `max(a,b)` or `a + b`. If it is `a + b` on a desktop with many idle cores, it will not
   be `max` on a phone with two performance cores.
2. **W4's artifact sizes forcing the §10.4 consolidation.** The moment consolidation is
   chosen, the shared-inference-queue risk is no longer a risk — A12 is dead, and W5 should
   be replanned at that moment rather than at the first on-device measurement.
3. **W4's exported single-model timings already at or above the S3 = 90 ms / S4 = 150 ms
   allocations** when scaled to the reference device. The concurrent budget has no room to
   absorb a stage that is over before contention is added.

**Secondary:** N9's 150 MB delivered install size, breached by two sets of native
libraries rather than by the models. `docs/artifact-budget.md` says plainly that nobody has
measured the combined native library size. Tripwire: the first `bundletool` output arriving
after the app is feature-complete rather than after the first successful build.

---

## W6 — CI/CD/CT

**Owning roles:** `devops` (workflows, required checks, artifact-size gates), `ml-engineer`
(the training step of the CT pipeline), `evaluation` (the promotion *criterion*).

> **Role gap — this one is real and needs a decision.** Nobody owns **promotion**.
> `evaluation` defines what "good enough" means but **has no `Edit` tool** and cannot write
> the gate. `ml-engineer` can write it, but it is the role whose model is being promoted,
> and a role that both sets the bar and reports clearing it will report clearing it — the
> exact separation `CLAUDE.md` calls load-bearing. `devops` owns the workflow file but not
> the criterion. Two branches, and this document does not pick one:
>
> | Branch | Consequence |
> |---|---|
> | **Create `.claude/agents/release-manager.md`** | A new role owning `docs/promotion-policy.md` and the gate job. One more role to introduce, and the course's role list grows |
> | **Split it: criterion as data, execution as code** | `evaluation` writes `docs/promotion-policy.md` as a machine-readable thresholds file (which its `Write` tool permits); `devops` writes a workflow that *reads* it and cannot alter it. No new role, and the separation survives because the thresholds live outside the workflow |
>
> The second is cheaper and preserves the boundary. It is still a decision, and it belongs
> to whoever owns the role definitions.

### Deliverables

| Artifact | Exists or does not |
|---|---|
| `.github/workflows/ci.yml` | ruff → `mypy --strict src` → `pytest -m "not integration"` → `tsc --noEmit` → the TS fusion suite → export parity → artifact-size check. Cheap checks first; nothing that compiles native code before a formatting error would have failed |
| `.github/workflows/ct.yml` | Dataset version in → train → evaluate → register → gate |
| `docs/promotion-policy.md` | The thresholds, machine-readable: mAP@50 floor, **per-class floor**, N4 ordering floor, combined artifact size, parity tolerance — plus the human step |
| The **contract-diff check** | A CI job that fails when `model_contract.json`'s label order or tensor layout changes without an accompanying ADR |
| The **artifact-size regression job** | Fails the PR when the bundled model total exceeds N9a |
| Required-checks configuration | Recorded in the repo (ruleset JSON or an equivalent), satisfying N18 |
| `docs/credit-budget.md` | Final reconciliation. Lifetime spend and final `Remaining` |

### Entry conditions

1. W4's tests pass locally and are markered correctly — `pytest -m "not integration"` runs
   with no GPU, no network, and no weights, or CI cannot run it.
2. **MLflow hosting is decided.** This is the hard one: a local file store is not reachable
   from a GitHub Actions runner, so a promotion job would have no registry to read. See
   Blocked on decisions.
3. A promotion criterion is agreed and its owner is named (the role gap above).
4. `docs/credit-budget.md` `Remaining` ≥ the CT dry-run estimate, with the estimate row
   written *before* the run.
5. A decision on whether CI includes a native build. `.claude/agents/devops.md` notes that
   export, parity, and artifact-size checks run on a plain Linux runner but that **building
   the iOS app needs macOS**, which is a cost. N18 says "every PR passes CI" and does not
   say whether CI includes an iOS build. Flagged below.

### Exit criteria

- [ ] A pull request shows all required checks green, and the list includes, by name:
      `ruff check`, `mypy --strict src`, `pytest -m "not integration"` with coverage ≥ 80%,
      `tsc --noEmit`, the TypeScript fusion suite, `pytest -m integration -k parity`, and
      the artifact-size check.
- [ ] **The contract-diff check is demonstrated, not asserted:** a branch that reorders two
      entries in `model_contract.json`'s `labels` array **fails CI**, and the run URL is
      recorded in `docs/promotion-policy.md`. A guard nobody has seen fire is a guard nobody
      knows is wired up.
- [ ] One CT run completes end to end from a dataset version input and **either** registers
      a new model version **or** records a refusal, in both cases logging the deciding
      metric, its value, and the threshold it was compared against from
      `docs/promotion-policy.md`. A refusal is a passing exit criterion — the gate working
      is the deliverable, not a promotion.
- [ ] `docs/promotion-policy.md` is machine-read by the workflow, not duplicated in it.
      Changing a threshold in the document changes the gate's behaviour; a `grep` of the
      workflow for a hardcoded numeric threshold returns nothing.
- [ ] N18's required-checks setting is recorded in the repository, not only in the GitHub
      UI.
- [ ] `docs/credit-budget.md`: lifetime spend ≤ **20.0**, final `Remaining` recorded and
      reconciled against the platform usage page on a stated date. Every divergence between
      an estimate and an actual has a note in the Notes table — the divergences are the
      finding, and this is the phase that reads them back.

### Dependencies

All four earlier phases. Specifically W3 for the registry the promotion step reads and W4
for the parity test CI runs as a gate.

### Risk most likely to derail this phase

**The CT dry run is unaffordable because the buffer was consumed upstream.**

The lifetime cap is 20 credits with **no top-up**, and `docs/credit-budget.md` states the
consequence in its own words: "If you run out in Week 4, Week 6 does not happen." W6 is
allocated 4.0 for one end-to-end CT dry run. It is also the last phase, so it absorbs every
overrun before it: a regenerated dataset version in W2, a re-run of hosted training in W3,
an Auto Label pass that went past 100 images. W4 and W5 spend nothing, so the entire
exposure lands between W2 and W3 and is invisible until W6 tries to run. The failure mode
is a `ct.yml` that is complete, reviewed, and has never executed — which is indistinguishable
from a broken one.

**Early warning sign:** the `Remaining` row of `docs/credit-budget.md` reading **below 6.0
at the end of W3**. Below 6.0 there is one CT run and no retry; below 4.0 there is no CT run
at all. Check it at the W3 exit gate, not at the W6 entry gate — at the W6 entry gate the
information is no longer actionable.

**Secondary:** N18's scope. If CI must include an iOS build, it needs macOS runners, which
bill at a multiplier and which nobody has costed. If it must not, then "every PR passes CI"
means "every PR passes the Linux subset," and the native build is a manual step somebody
will skip. Either answer is fine; the absence of one is not.

---

## Blocked on decisions

Four decisions are recorded as open in `docs/requirements.md` and `docs/artifact-budget.md`.
That is the authoritative list, and it is reproduced first. A second table follows for
places where the requirements are **silent** and this plan is nonetheless forced to choose —
those are flags, not decisions, but each one becomes a decision the moment somebody writes
code against it.

### The four open decisions

| # | Decision | Blocks | Latest point it must be resolved | Who decides |
|---|---|---|---|---|
| 1 | **Object taxonomy** | W2 (its first deliverable), and through label order, W4's C11 and W5's overlay. N3 has no target without it | **Before Roboflow dataset version 1 is generated in W2.** A version generated first must be regenerated, which bills again and invalidates every count in the dataset card | Product. `dataset-engineer` proposes; a human accepts the ADR |
| 2 | **MLflow hosting** | N19 nominally in W3; **hard-blocks W6's promotion logic** | **Before W6's `ct.yml` is written**, absolutely. **Strongly preferred before W3's first tracked run**, because runs recorded against one backend must otherwise be migrated | Project team |
| 3 | **Device floor** | The interpretation of W5's N1/N2/N8 measurements, and — via §6.2's escape levers — W4's choice of `DEPTH_SIZE` and YOLO11 variant | **A candidate before W5 begins** (so the right handset is measured); **confirmed by W5's peak-RSS measurement**. See the split below | Product, informed by W5's measurement |
| 4 | **One runtime or two** | W4's export targets and W5's entire runtime layer. **It is the one that reaches the latency budget** (§6.2, §10.4) | **At W4's first artifact-size measurement** — that measurement is what decides it. It cannot be deferred past W4's exit: consolidating is a real export workstream, and discovering the need for it in W5 puts an export phase inside the mobile phase | Project team, on W4's measured numbers |

**Decision 2, by branch** (architecture §10.3, reproduced with its W6 consequence):

| Branch | W3 | W6 |
|---|---|---|
| Local file store | Works. Runs are not shareable; N19 satisfied for one person | **Breaks.** A GitHub Actions runner cannot read a local `mlruns/`. The promotion gate has no registry to query |
| Self-hosted server | Works. A service to run — and one that must never appear on the inference path | Works, if the runner can reach it |
| Managed | Works. An account and a bill **outside the 20-credit ledger** | Works |

The `RunTracker` protocol (architecture §8.4, C13) is what keeps this from blocking W3 on
day one — no module names a backend, and `NullTracker` covers the tests. It does **not**
keep it from blocking W6.

**Decision 3 is circular unless it is split, so split it.** The floor is decided *by* the
peak-RSS measurement, and the measurement needs a device. Resolve as two steps: name a
**candidate floor** before W5 (the runtimes' own floor is iOS 17+ / Android 13+ per
`docs/artifact-budget.md`, which is a starting point, not an answer); measure peak RSS on
it; **confirm or raise** the floor from the measurement. If the confirmed floor lands below
the iPhone 13 / Pixel 6a reference, architecture §6's entire allocation table is re-derived
and N1 may not survive — which is a W5 finding with W4 consequences, and is why the
candidate cannot wait until W5 is underway.

### A fifth, not yet on the requirements list

| # | Decision | Blocks | Latest point |
|---|---|---|---|
| 5 | **Where the N4 depth-ordering pair set lives** | W3's `docs/model-cards/depth-v1.md`, and therefore W4's ability to detect that quantization broke depth ordering | **Before W2 exits.** It is a W2 deliverable with no defined home |

Roboflow has no pairwise-ordering annotation type, `data/` is gitignored and assistants must
not modify it, and MLflow stores run artifacts rather than versioned evaluation ground
truth. The candidate branches, none of which this document picks:

| Branch | Consequence |
|---|---|
| A JSON file under `tests/fixtures/depth_ordering/`, committed | Small, reviewable, diffable, offline-testable. Couples evaluation ground truth to the source repo, which the project otherwise avoids |
| A Roboflow tag or attribute convention on the test split | Keeps it versioned with the dataset. Bends an annotation type to a purpose it was not built for, and reading it back may bill |
| Re-scope N4 | Measure ordering against a proxy — for example, agreement between the fp32 reference and the exported artifact, which tests the *export* but not the *model* | 

This needs an ADR. It is listed here rather than in `docs/requirements.md` because
requirements state N4's target and are silent on where its ground truth comes from.

### Requirements are silent, and the plan is forced to choose

These are flags. Each one is a decision that will be made by whoever writes the code first,
in a commit message nobody reads, unless it is made deliberately.

| Question | Phase it lands in | Where it is already noted |
|---|---|---|
| Does F3's "source image frame" mean the file's pixel grid or the decoded working image? | W4 (fusion), W5 (overlay) | Architecture §3 and §12 — resolved provisionally as the decoded working image, with `decode_scale` carried so it is reversible |
| Confidence and NMS IoU thresholds — they move N3 directly | W3 (measured with them), W4 (baked into C11) | Architecture A8 |
| `max_detections` per image | W4 (bounds S5 and S6's latency allocations) | Architecture A7 |
| What the UI does with a `null` `relative_depth` on **one** object — F8 covers only the zero-detections case | W5 | Architecture §12 |
| Does CI include a native iOS build? N18 says "every PR passes CI" and does not scope it | W6 | `.claude/agents/devops.md`, method 4 |
| Who owns the promotion criterion, given `evaluation` has no `Edit` | W6 | The role gap in W6 above |
| Is the 2.0-credit "Lesson 04 — Backend Engineering" allocation returned to the buffer? | W4 | The flag under [Phase map](#phase-map) |
| Does `.claude/agents/export.md` get created, or does `CLAUDE.md`'s role table get corrected? | W4 | The role gap in W4 above |

---

## What this plan deliberately does not contain

| Absent | Why |
|---|---|
| Dates and durations | Requested, and correct: the dependency graph is the plan. A slipped date says nothing about what became unsafe to start |
| A resolution to any of the five open decisions | Each is stated with its branches and its deadline. An architecture document that closes an open decision by assumption is worse than one that leaves it open |
| A server, a deployment, or a rollback plan | ADR 0001. "Ship" here is an app binary and an app-store review |
| A batching or throughput workstream | F6 is Won't |
| Contingency phases | The escape levers are named where they bite — §6.2's three for latency, §10.4's two for size. A named lever inside the phase that needs it beats a contingency phase nobody schedules |
