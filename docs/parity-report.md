# Export parity report — the device against the reference

> The question this answers: **is the app running a model that still behaves like the one
> you trained?** Not "does it run", and not "does it produce plausible output".

## What was compared

| | Path A (the app) | Path B (reference) |
|---|---|---|
| Where | iPhone 17 Simulator, iOS 27.0, UDID `DACAF190-5F03-4001-89BE-8F336DCDCB15` | Python, this machine |
| Weights — detection | `app/assets/models/detector.tflite` (unmodified; verified at the end of this report) | `runs/detect/runs/v1-baseline/weights/best.pt` (original PyTorch checkpoint) |
| Weights — depth | `app/assets/models/depth.pte` | `depth-anything/Depth-Anything-V2-Small-hf` (original HF checkpoint, eager) |
| Runtime — detection | `react-native-fast-tflite@3.0.1` / `TensorFlowLiteC 2.17.0` | `ultralytics.YOLO(...).predict(...)` |
| Runtime — depth | `react-native-executorch@0.10.7` (real `execute()` call) | `transformers.AutoModelForDepthEstimation`, eager |
| Image | `kv2_align_kv2_2014-12-18_11-47-06_260595134347_0000037.jpg`, 730×530 | same |

> **Path B uses the PyTorch/HF weights directly, not the exported artifacts.** Detection
> went through `ultralytics.YOLO(best.pt).predict(...)`; depth went through
> `transformers.AutoModelForDepthEstimation.from_pretrained(...)` eager — the exact loading
> pattern `src/smart_scene_analyzer/export/executorch.py`'s `_load_depth_wrapper` already
> uses, and `tests/test_export_parity.py`'s `_run_pytorch` for detection. Neither reference
> touches `detector.tflite` or `depth.pte`.

**Why this image:** it is the original, full-frame photo this project's bundled test
images are derived from — `app/assets/test-images/round4-landscape-4x3.jpg`,
`round4-portrait-4x3.jpg`, and `round4-portrait-16x9.jpg` are crops/rotations of this same
office-cabinet scene, built for an earlier lesson's aspect-ratio geometry check, not
independent content. It contains two in-taxonomy classes in a natural layout (a chair, a
storage cabinet) rather than being synthetic. It does **not** exercise: a scene with zero
objects, a crowded scene with many overlapping boxes, outdoor lighting, or any class other
than `chair`/`storage`.

## On-device detection: blocked, independently re-confirmed this round

`react-native-fast-tflite`'s `Invoke()` on `detector.tflite` has been failing all session
(documented in `app/src/inference/decode.ts` and `app/src/pipeline/analyzeImage.ts`). A
precise root-cause diagnosis — the FPN's `RESIZE_NEAREST_NEIGHBOR` op is encoded as
operator version 3 (`half_pixel_centers=True`), which `TensorFlowLiteC 2.17.0` (the
version the app's podspec pins, with no flexibility) apparently cannot execute — was
proposed this session along with a byte-level patch downgrading it to version 1.

**This round tested that patch on-device, and it did not fix `Invoke()`.** Full
methodology, evidence, and the exact on-screen failure text are in
[`docs/parity-evidence/round1-ondevice-patch-test.md`](parity-evidence/round1-ondevice-patch-test.md).
Summary: Python verification of the patch (via `ai_edge_litert`) turned out to be weaker
evidence than it first appeared — the **original, unpatched** file also loads and runs
fine under `ai_edge_litert`, so that runtime was never actually sensitive to the
difference the patch addresses. On-device, under the real `TensorFlowLiteC 2.17.0`
runtime, both the original and the patched file fail `Invoke()` identically. The
hypothesis is **refuted**, not confirmed; the true blocker remains unidentified beyond
"structural/toolchain, not data-dependent" (a trivial all-zero input fails exactly like a
real image).

**Consequence for this report:** there is no real on-device detection output for any
image, this one included. The detections below are the PyTorch reference's own, reused on
the app side only to give the depth-ordering check (next section) real object locations
to test against — they are **not** the app's own detections, and are labeled as such in
`docs/parity-evidence/app-result.json`.

## Detections — Python reference (Path B)

| # | Class | Confidence | Box (source px, 730×530) |
|---|---|---|---|
| 1 | chair | 0.916 | (156.5, 102.1) – (278.9, 302.1) |
| 2 | storage | 0.850 | (257.0, 4.9) – (564.6, 496.2) |
| 3 | chair | 0.321 | (0.0, 224.9) – (55.1, 524.6) |

No app-side (Path A) detection output exists to compare — see the blocked-detection
section above. **Box/class/confidence parity is not measurable this round**, which is
itself the finding for the detection half of this report, not a gap silently left blank.

## Depth

Depth is compared by **ordering**, not by value — the output has no scale, so comparing
magnitudes across two runtimes is meaningless. Because there is no real on-device
detection, the three object locations above (Path B's real detections) were reused as the
sample points for **both** sides: the project's own reference reduction,
`src/smart_scene_analyzer/fusion.py`'s `fuse()` (median over an inset box, the same
algorithm `app/src/fusion/index.ts` ports), was run once against the Python eager depth
map and once against the **real, on-device** `depth.pte` output — captured via a temporary
dev-only dump (`runDepthInference` on the real decoded image, real `execute()` call; see
the patch-test note for why console output wasn't reachable and what captured it instead).
Raw evidence: `docs/parity-evidence/{python-reference,app-result,app-depth-raw}.json`.

| Object | relative_depth (Python eager) | relative_depth (app, on-device) | Rank in B (near→far) | Rank in A | Match? |
|---|---|---|---|---|---|
| storage | 3.355 | 3.329 | 0 | 0 | yes |
| chair (small, corner) | 2.341 | 2.278 | 1 | 1 | yes |
| chair (large) | 0.461 | 0.561 | 2 | 2 | yes |

**Ordering preserved: yes, 3/3.** The small corner chair's box also independently landed
`depth_quality: "mixed_region"` on **both** sides (its region straddles a real depth
discontinuity — the box is partly background) — the same reduction flagging the same
unreliability on both the reference and the on-device run is corroborating evidence the
port/quantization is behaving consistently, not coincidentally agreeing.

## Differences, classified

| Difference | Category | Distinguishing test | Result |
|---|---|---|---|
| No on-device detection output at all (any image) | **Not one of the four** — a prerequisite toolchain/runtime-compatibility failure, upstream of where preprocessing/decoding/quantization/port differences could even be observed | See below | Confirmed structural, not data-dependent |
| On-device depth values differ from PyTorch eager by a small, bounded amount at matched object locations | **Quantization** | Does the int8 export's whole-map rank correlation against eager fall within the project's already-measured noise floor? | Yes — consistent with, not worse than, the existing floor |

**Why the detection blocker isn't one of the four**, each ruled out with its own test:
- *Not quantization* — `tests/test_export_parity.py::test_detection_parity_per_class`
  (Python `ai_edge_litert` running the actual exported file, no on-device runtime
  involved) shows the export's **content** agrees with PyTorch at a 91.9% match rate
  (34/37), worst class delta −0.133 — ordinary int8 drift, not a broken export.
- *Not decoding* — decoding only runs on a tensor `Invoke()` produced; `Invoke()` never
  returns one.
- *Not preprocessing* — the failure reproduces identically on a synthetic all-zero
  `[1,3,640,640]` tensor (`DEV_DEBUG_INVOKE`), which has no letterbox or normalization to
  get wrong.
- *Not the port* — there is no TS-vs-Python disagreement to observe; no raw tensor ever
  reaches either decoder.
- Positively confirmed as a runtime/toolchain issue: `AllocateTensors` succeeds (the op
  graph resolves) but `Invoke()` fails generically, on both the original and the
  op-version-patched file, under the real pinned `TensorFlowLiteC 2.17.0`. See
  `docs/parity-evidence/round1-ondevice-patch-test.md` for the full test.

**Depth quantization evidence**, beyond this round's own 3/3 object-level rank match:
`tests/test_export_parity.py::test_depth_export_parity` (pre-existing, re-run this round)
measures the exported `.pte`'s whole-map rank correlation against PyTorch eager at
**mean 0.9447, min 0.8748, n=8** held-out images — a floor set from measured data, not
fit to this report. This round's object-level deltas (relative_depth differing by
roughly 10–20% in magnitude, ranks identical) are consistent with that floor, not evidence
of a new problem.

Reference for the four categories:

| Category | Signature | Test |
|---|---|---|
| **Preprocessing** | Boxes systematically offset or scaled | Does the offset scale with the letterbox padding? |
| **Decoding** | Boxes plausible, classes wrong or confidences oddly distributed | Is it a constant class-index shift? |
| **Quantization** | One or two classes degraded, rest fine | Does an fp32 export agree where int8 does not? |
| **The port** | Detections identical, depth values differ | Do the shared fusion fixtures still pass? |

On the port specifically: the Python fusion suite (`tests/test_fusion.py`, the reference
`fuse()` this report's own depth-ordering check calls directly) passes 29/29. The
TypeScript port (`app/src/fusion/index.ts`, `app/src/fusion/index.test.ts`) could **not**
be independently re-run this round — `app/` has no JS test runner wired up yet (no `jest`
or equivalent in `devDependencies`, no config, confirmed by attempting to run it). This
report relies on the real on-device pipeline (`analyzeImage.ts`) exercising the same port
code path for its depth numbers, not on a passing test suite — flagged under "what was
not tested" below.

## Tolerances

| Quantity | Tolerance | Set before or after seeing the data? |
|---|---|---|
| Box IoU | 0.5 (COCO/Pascal-VOC convention) | **Before** — reused verbatim from `tests/test_export_parity.py`'s pre-existing `MATCH_IOU_THRESHOLD`, set in an earlier lesson, not chosen for this report. **Not exercised this round** — see justification. |
| Confidence delta | < 0.5 absolute | **Before** — reused verbatim from `tests/test_export_parity.py`'s pre-existing assertion. **Not exercised this round.** |
| Depth ordering | Exact match required | — (fixed by the prompt, not chosen) |

**Justification:** no on-device detection exists this round to apply a box/confidence
tolerance to, so none was set fresh — stating a number here would imply a real comparison
happened. The IoU and confidence-delta values above are carried over from this project's
own existing export-parity test (written before this report, against different data) so
that whenever on-device detection starts working, the same bar applies rather than one
quietly chosen to make a future result look good. Depth ordering's tolerance is exact
match (not a correlation threshold) because only three objects were available this round;
3/3 is reported as what it is — a small-n perfect result, not evidence of robustness at
scale (see "what was not tested").

## Verdict

**Not confirmed, and split by model.**

- **Depth:** the real on-device `depth.pte` run, compared against PyTorch eager at three
  real object locations, preserved near-to-far ordering exactly (3/3) with value deltas
  consistent with this project's own previously-measured int8 quantization floor
  (mean rank correlation 0.9447 across 8 images). For depth specifically, the device is
  behaving like the trained model, within already-accepted quantization noise — on the
  one image and three objects tested.
- **Detection:** **not confirmed, and currently not even measurable.** The app does not
  run real on-device detection at all; a precisely-targeted fix was tried this round and
  did not work. Nothing about this report says the detector's trained behavior survived
  export — only that the export file's *content* independently checks out
  (`test_detection_parity_per_class`, 91.9% match) while the *runtime* that would execute
  it on an iPhone cannot.

## What was not tested

- **Other images** — one image, three objects, is one sample. A busy scene, an empty
  wall, an outdoor shot, or a different class entirely (`bed`, `bin`, `door`, `lamp`,
  `pillow`, `screen`, `sofa`, `table`) were not exercised. The 3/3 depth-ordering result
  should not be read as "depth ordering always holds" — it is "depth ordering held once."
- **Other devices or backends** — tested only on the iOS Simulator (iPhone 17, iOS 27.0).
  A physical device, Android/NNAPI, and any GPU delegate are all untested; simulator
  numbers are not device numbers, and this report draws no conclusion about either.
- **The TypeScript fusion port's own test suite** — not independently re-run this round
  (no JS test runner configured in `app/`); this report leans on the real pipeline
  exercising that code, not on a green test run.
- **Scenes outside the training distribution** — not evaluated; this image is in-domain
  (the same source distribution as training/calibration data).
- **A working fix for on-device detection** — out of scope for this report by design. The
  patch tested this round was diagnostic only, confirmed not to work, and fully reverted;
  no fix is proposed here.

A reader should not conclude from this report that the app's detection path is close to
working, that the depth result generalizes beyond this one image, or that the op-version
hypothesis for the `Invoke()` failure remains live — it was tested and refuted.

## Artifact integrity

`app/assets/models/detector.tflite` is confirmed the original, unmodified export at the
end of this report:

```
sha256: 3f5bf68248e5c36abf4f2fb260c38f1ec27d76203a22cc61e054f510b93ad83d
```

— matching both `app/assets/models/model_contract.json`'s recorded hash and
`/tmp/detector_original_backup.tflite` (the pre-patch backup taken before this round's
diagnostic swap). `app/App.tsx` is confirmed reverted to its pre-report content (792
lines, re-read and compared). The simulator was rebuilt once more with the restored file
and reverted source, and a screenshot confirms the running app shows no leftover debug
output.
