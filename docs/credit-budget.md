# Credit budget

**Hard cap: 20 Roboflow credits for the entire project, Lessons 01 through 06.**

There is no top-up. If you run out in Week 4, Week 6 does not happen. This file is the
ledger, and it is the reason the assistant can answer "can I afford this?" before it
spends anything.

It is also machine-read. `.claude/hooks/credit_gate.py` parses the **Remaining** row and
the Ledger table, and refuses billed Roboflow tool calls until a row exists with an
estimate and no actual. If you change this file's shape, change the hook with it —
a parser that silently finds nothing reports a balance of zero and blocks everything.

| | |
|---|---|
| Workspace | `reynaldo-carrillo-enroutesystems-com` |
| Project ID | `smart-scene-analyzer-2026-09-08` (object-detection, Public Domain) |
| Project URL | https://app.roboflow.com/reynaldo-carrillo-enroutesystems-com/smart-scene-analyzer-2026-09-08 |
| Plan | Public (free) |
| Starting balance | 20.0 |
| **Remaining** | **0.0** |
| Last reconciled | 2026-09-28 — **against the platform**, which reported `creditsRemaining: 0` in an `INSUFFICIENT_CREDITS` error from `versions_generate`. The previous figure of 20.0 was this project's self-imposed cap, never a measured balance |

---

## Planned allocation

Set once, at the start of Lesson 02. Revise it deliberately, not silently.

| Lesson | Budget | What it buys |
|---|---|---|
| 01 — Project Definition | 0.0 | No platform contact |
| 02 — Dataset Engineering | 4.0 | Uploads, storage, two version generations, the 100-image Auto Label audit |
| 03 — Model Development | 3.0 | One hosted training run (Roboflow Instant is free) |
| 04 — Backend Engineering | 2.0 | One hosted-API latency and cost comparison |
| 05 — On-Device Inference & Delivery | 0.0 | The models run on the handset. The app has no credentials, so no Roboflow call on the inference path is even possible |
| 06 — CI/CD/CT | 4.0 | One end-to-end continuous-training dry run |
| Buffer | 7.0 | Re-runs, mistakes, a regenerated version |
| **Total** | **20.0** | |

---

## Ledger

Append one row per billed operation. Free operations do not need a row, but recording
the significant ones (the 4-image Auto Label preview, a Roboflow Instant run) is worth
doing — it documents that you took the free path deliberately.

| Date | Lesson | Operation | Rate | Estimated | Actual | Running total | Remaining |
|---|---|---|---|---|---|---|---|
| | | *starting balance — a self-imposed cap, never a measured platform balance (see Notes, 2026-09-28)* | | | | 0.0 | 20.0 |
| 2026-09-10 | 02 | Upload 8,332 SUN RGB-D RGB images (`image/` only, unannotated) + first month storage | 1cr/10,000 uploaded; 1cr/5,000 stored/month | 2.50 | ⚠️ UNRECONCILED | 2.50 | — |
| 2026-09-28 | 02 | Generate dataset version 1 — 7,724 source images, full Lesson 02 config | 1cr/20,000 images | 0.93 | **0.00** | 2.50 | 0.0 |

---

## Rates

Verify against `roboflow:plans-and-pricing` before relying on these. Upstream rates
change, and the skill is read fresh from disk every session; this table is not.

**What 1 credit buys**

| Operation | 1 credit |
|---|---|
| Uploads | 10,000 images |
| Storage | 5,000 images/month |
| Version generation | 20,000 images |
| Auto Label | 100 images |
| Model training (GPU) | 30 minutes |
| Hosted serverless inference (v2) | 500 seconds of execution |
| Self-hosted inference | 3,000 images |
| Dedicated deployment (GPU) | 1 hour of **uptime** |
| Batch processing (GPU) | 15 minutes |

**Free**

- Auto Label "Generate Test Results" on a 4-image subset — unlimited, and the whole
  reason prompt tuning is cheap
- Roboflow Instant training — few-shot, object detection only
- Universe search and browsing; RoboQL queries; tags; splits
- Workflow authoring, block listing, and spec validation — only *running* a workflow bills
- Cancelling a training run early ("refund if early in training")

---

## Reconciliation

At the end of every lesson:

1. Open `app.roboflow.com/<workspace>/settings/usage`.
2. Compare the platform's total against this ledger's running total.
3. Update **Remaining** and **Last reconciled** above.

**A gap between your estimate and the actual charge is a finding, not a mistake.** It
means an operation costs something you did not model. Write down what it was — that note
is worth more than the credit it cost you.

---

## Notes

Record anything that surprised you: an operation that billed unexpectedly, a rate that
turned out to be different from the table, a free-tier limit you hit.

| Date | Note |
|---|---|
| 2026-09-08 | `projects_create` billed **0 credits** — no rate exists for project creation, and `credit_gate.py` is deliberately scoped to `trainings_create\|models_infer\|versions_generate\|workflows_run`, so it does not gate this call. Recorded as a free operation taken deliberately; no Ledger row, because a row with an estimate and a blank Actual reads as *pending* to the gate. |
| 2026-09-08 | Project named with a date suffix (`-2026-09-08`) rather than the bare `smart-scene-analyzer` the lesson uses. Roboflow appended no random suffix, so the ID equals the display name. Lesson 02 steps 9 and 15 were changed to take a `<project-id>` placeholder read from this file rather than a hardcoded slug. |
| 2026-10-01 | Ledger rows closed out. The **version-generation row is reconciled at `Actual` 0.00** — the call was *refused* with `INSUFFICIENT_CREDITS`, so nothing billed. It had been sitting with an estimate and a blank actual, which `credit_gate.py` reads as a **standing authorization**: a 0.93-credit approval for an operation that never ran, which would have waved through the next billed call. A refused operation must be closed at zero, not left pending. |
| 2026-10-01 | The **upload row's `Actual` is marked UNRECONCILED rather than filled in.** Its true cost exists only at `app.roboflow.com/reynaldo-carrillo-enroutesystems-com/settings/usage`; the credit-ledger skill forbids copying the estimate across, since doing so defeats the entire mechanism. `Remaining` is left as `—` on that row because the figure is unknown, not because it is zero. This is the one entry that would have caught the zero balance before a version generation was planned against it. |
| 2026-09-28 | **The platform reports 0 credits remaining.** Discovered only when `versions_generate` refused with `INSUFFICIENT_CREDITS` (`creditsNeeded: 0.65655`, `creditsRemaining: 0`). The ledger had read 20.0 since the project began. That number was this project's self-imposed cap, not a balance read from the platform, and because `Actual` was never reconciled on any row, nothing ever contradicted it. **The reconciliation step is not bookkeeping — it is the only thing that would have caught this**, and deferring it is what let a version generation be planned, estimated, approved and attempted against a balance of zero. |
| 2026-09-28 | Version generation bills on the **post-augmentation** image count — confirmed, not assumed. The refusal quoted 13,131 images and `creditsNeeded: 0.65655`; `13,131 / 20,000 = 0.65655` exactly. Lesson 02 step 13 records this as undocumented and asks which basis was assumed: it is post-augmentation. Corollary: the **default max version size is 2×** (5,407 train × 2 + 1,545 + 772 = 13,131), not 1× and not 3×. |
| 2026-09-28 | The full Lesson 02 augmentation config **is** expressible through the MCP `versions_generate` tool. A request carrying `resize.format: "Fit within"`, `brightness`, `exposure`, `blur` and `noise` passed validation and failed only at the credit check. The API validates strictly — an incomplete `flip` (missing `vertical`) was rejected outright — so silent key-dropping is not a risk here. Earlier caution about undocumented keys was warranted but, in the event, unnecessary. |
| 2026-09-10 | Upload completed: 8,332/8,332 images, 0 duplicates, 0 errors, CLI exit 0; platform reports `unannotated: 8332`, `images: 0`. **`Actual` is still blank** — it must be read from `app.roboflow.com/reynaldo-carrillo-enroutesystems-com/settings/usage`, not copied from the estimate. While that Ledger row carries an estimate and no actual, `credit_gate.py` reads it as a *standing* pending authorization and will let the next billed MCP call through. Close it at the next reconciliation. |
| 2026-09-10 | Scope decision: uploaded `image/` (RGB) only — 8,332 of the 35,691 image files in `data/`. Excluded `depth/` and `depth_bfx/` (Roboflow stores no depth ground truth) and `fullres/` (alternate resolutions of a 4,850-scene subset; different content hashes, so no dedupe against `image/`). All four would have cost 3.57 one-time and **7.14/month** storage, exhausting the 20-credit cap inside three months. |
| 2026-09-10 | SUN RGB-D reuses filenames across scene directories: the 8,332 RGB files hold only 3,417 distinct basenames. A naive flatten into one upload folder silently overwrites 4,915 of them and uploads 41% of the dataset with no error and exit 0. Staged instead as symlinks with path-derived names (`/tmp/ssa-upload-8332`, manifest `/tmp/ssa-upload-manifest.csv`) and verified 1:1 before uploading. |
| 2026-09-10 | Annotation conversion cost **0 credits** — done locally from `SUNRGBDMeta2DBB_v2.mat`, deliberately instead of Auto Label. Auto Label on 8,332 images at 1cr/100 would have been 83.3 credits, over 4× the entire lifetime budget. 51,019 boxes across 7,724 images for free. |
| 2026-09-10 | `data/` lost ~2,003 scene directories between 2026-09-08 and 2026-09-10 (45,706 → 35,691 image files). No process was modifying it; the change happened between sessions. Counted from the live tree immediately before staging rather than trusting the earlier figure. |
