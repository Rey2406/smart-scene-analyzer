# Requirements — Smart Scene Analyzer

Filled in at Lesson 01 step 8. The inference target is **on-device** (ADR
[0001](decisions/0001-inference-target.md)), and that decision reshapes this document:
the worksheet this was copied from assumes a served API, so several rows below are
answered as *not applicable* with the reason stated, and replaced by the on-device
question that actually constrains the design.

A row marked **N/A — no service** is not an unanswered row. It is a row whose absence is a
consequence of the architecture, and saying so is the point.

## Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| F1 | The app accepts a single RGB image — captured or picked — and returns detected objects | Must |
| F2 | Each detection carries a class label and a confidence score | Must |
| F3 | Each detection carries a bounding box in `xyxy`, absolute pixels of the **source image** frame | Must |
| F4 | Each detection carries a **relative depth score** — relative inverse depth, larger is nearer, no unit, comparable only within one image | Must |
| F5 | The app produces a dense depth map for the whole image | Must — it is the input F4 is derived from. Whether it is *displayed* is F9 |
| F6 | The app accepts a batch of images in one operation | Won't — a handset UI analyses one image at a time; batching is a server affordance with no user here |
| F7 | The result reports the model version used, for both models | Must |
| F8 | Zero detections renders as a successful empty result, visually distinct from "loading" and from "the model failed to load" | Must |
| F9 | The depth map is viewable as a shaded overlay, ordered but unlabelled by distance | Should |
| F10 | The app runs with the device in airplane mode, with no degradation | Must — this is the architecture's observable claim, so it is a requirement and a test |

### Object taxonomy

Pending — see Lesson 02. The class list is an **open decision**; it is fixed in Lesson 02
and recorded in `docs/taxonomy.md`, which does not exist yet.

This blocks N3: an mAP target over an unknown number of classes is not a target. The
provisional figure in N3 is marked as such.

### Out of scope

- **Metric-scale absolute depth.** Depth Anything V2 returns relative inverse depth with
  no scale. The system cannot state a distance, and no part of the UI may imply one. This
  is enforced by `.claude/hooks/units_guard.py`, not merely intended
- Instance segmentation — boxes only
- Video, temporal tracking, and frame-to-frame identity
- Multi-image reconstruction, stereo, or any use of camera intrinsics
- A server, an account, a login, or any synchronization between devices
- Model updates without an app-store release — a consequence of ADR 0001, named here so
  nobody rediscovers it as a surprise

---

## Non-functional requirements

### Performance

| ID | Requirement | Target | How it is measured |
|---|---|---|---|
| N1 | End-to-end p95 latency, single 1280×720 image | **≤ 400 ms** | On-device, release build, **model load excluded** (that is N6). Capture/pick to drawn overlay. Reference devices below |
| N1a | Bytes leaving the device per analysis | **0** | Airplane-mode run produces an identical result (F10) |
| N2 | p50 latency, same image | **≤ 250 ms** | Same conditions as N1 |
| N3 | Detection accuracy | mAP@50 ≥ **0.50** *(provisional)* | Held-out test split, dataset version stated. Provisional until the taxonomy is fixed in Lesson 02 |
| N4 | Depth quality | Pairwise **depth-ordering accuracy ≥ 0.90** over annotated object pairs | Held-out test split. See the note below on why this is not δ<1.25 |
| N5 | Sustained throughput | ≥ **20 consecutive analyses** with no thermal throttling and no memory growth | Repeated-run instrument on a reference device |
| N6 | Cold model load, both models | ≤ **3.0 s** | First analysis after app launch. The user is watching a spinner for this |

> **Where N1 comes from.** Working backwards from the interaction: a person taps a button
> and waits for boxes to appear. Under roughly 400 ms that reads as responsive. The number
> is a budget the Architecture Agent allocates across letterboxing, two inferences, fusion,
> and drawing — set from the experience, not from what the models happen to do.

> **N1 names its device, because on-device latency is meaningless without one.** Reference
> devices: **iPhone 13 (A15)** for iOS and **Pixel 6a (Tensor G1)** for Android, release
> builds, screen on, battery above 50%, not thermally throttled.
>
> ⚠️ These are the *measurement* reference, not the supported floor. **The device floor is
> an open decision** — see `docs/artifact-budget.md`. If the floor lands below these, N1
> must be re-measured on the floor device and this target may not survive contact.

> **There is no network transit line, and that is the whole point of ADR 0001.** The
> worksheet's warning about upload size at a given link speed — 250 KB at 5 Mbit/s spending
> 400 ms before the server sees a byte — describes the term this architecture deleted. The
> budget is now entirely compute the device performs, which is why N1's device matters as
> much as a link speed used to.

> **Why N4 is not δ<1.25.** That metric compares predicted depth to ground truth in metric
> units, and this system has no metric depth to compare — applying it would require fitting
> a scale and shift per image, which invents the very quantity the project refuses to
> claim. What the product actually relies on is **ordering**: that a nearer object scores
> as nearer. So that is what is measured. A metric already assumes an answer to the
> question the architecture declined to ask.

### Resource constraints

| ID | Requirement | Target |
|---|---|---|
| N7 | Max input image accepted | Any the camera or picker returns; downscaled to the model input on entry. Long edge ≥ 3024 px handled without an out-of-memory failure |
| N8 | Peak RSS, both models loaded and one analysis in flight | ≤ **1.2 GB** — the figure that decides which phones are excluded. Tracked in `docs/artifact-budget.md` |
| N9 | Delivered install size, per platform | ≤ **150 MB** — measured as *delivered*, from store reporting or `bundletool build-apks --connected-device`, with the method stated |
| N9a | Bundled model artifacts, combined | ≤ **40 MB** |
| N10 | GPU or accelerator required? | No. The app must run on CPU. A delegate (Core ML, NNAPI, GPU) may be used when present and must be a measured speedup, never a correctness dependency |

### Reliability and operations

| ID | Requirement | Target |
|---|---|---|
| N11 | Availability | N/A — no service. The functional equivalent is F10: the app works offline, by construction rather than by uptime |
| N12 | Behaviour when a model fails to load | **Degrade to detection-only** if depth fails; fail visibly if detection fails. Never render a partial result that looks complete. The fusion layer must therefore tolerate a missing depth branch — this is a design constraint, not error handling |
| N13 | Behaviour on malformed or oversized input | N/A — no HTTP. An in-app error state naming what failed; never a silent empty result, which F8 reserves for genuine zero detections |
| N14 | Are images retained? | **No.** Not uploaded, not written to app storage, not logged. Structural: there is nowhere to send them |
| N15 | Structured logging | Development builds only. **No telemetry on the inference path** — not even non-blocking, which would put a radio on a path this architecture guarantees is offline |

### Development quality gates

| ID | Requirement | Target |
|---|---|---|
| N16 | Test coverage on `src/` | ≥ **80%** |
| N17 | Type checking | `mypy --strict` passes on `src/`; `tsc --noEmit` passes on `app/` |
| N18 | Every PR passes CI before merge | Yes |
| N19 | Reproducible training runs | Seed, dataset version, and config recorded in MLflow |
| N20 | Export parity | The exported artifact and the PyTorch reference agree on shared fixtures within a stated tolerance, as a test |
| N21 | Fusion port parity | `app/src/fusion/` reproduces `src/smart_scene_analyzer/fusion.py` on the **same** fixture set |

> N20 and N21 are quality gates rather than nice-to-haves because they are the only
> mechanism by which two implementations of one algorithm, in two languages, stay honest.
> Without them "the app disagrees with the reference" is a debate rather than a test.

---

## Open decisions blocking this document

| Decision | Blocks | Status | ADR |
|---|---|---|---|
| Inference target: cloud vs. on-device | N1, N7–N10, the entire architecture | **decided** — on-device | [`0001-inference-target.md`](decisions/0001-inference-target.md) |
| MLflow hosting | N19 | open | — |
| Device floor (minimum supported handset) | N1, N2, N8, and whether two runtimes fit at all | open — see `docs/artifact-budget.md` | — |
| Object taxonomy | The F-list, N3, all of Lesson 02 | open — Lesson 02 | — |
| One runtime or two | N8, N9, N9a | open — the fallback is consolidating YOLO11 to `.pte` or Depth Anything V2 to TFLite | — |

> The worksheet's row **"N1: client- or server-observed"** is resolved by ADR 0001 rather
> than answered: with no network on the inference path there is no client/server split to
> observe from either side. Its replacement is the **device floor** row, which is the
> question on-device latency actually depends on.

Resolve each with `/adr <title>` and link the resulting file here.
