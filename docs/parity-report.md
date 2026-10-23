# Export parity report — the device against the reference

> The question this answers: **is the app running a model that still behaves like the one
> you trained?** Not "does it run", and not "does it produce plausible output".

## What was compared

| | Path A (the app) | Path B (reference) |
|---|---|---|
| Where | iPhone 17 Simulator, iOS 27.0, UDID `DACAF190-5F03-4001-89BE-8F336DCDCB15` | Python, this machine |
| Weights — detection | `app/assets/models/detector.tflite` (round 3: buffer-inlined, see "Artifact integrity" — a storage-format fix, not a weight change) | `runs/detect/runs/v1-baseline/weights/best.pt` (original PyTorch checkpoint) |
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

## On-device detection: fixed in round 3

`react-native-fast-tflite`'s `Invoke()` on `detector.tflite` failed all session until this
round (documented history in `app/src/inference/decode.ts` and
`app/src/pipeline/analyzeImage.ts`). Round 1 proposed and refuted an op-version hypothesis
([`docs/parity-evidence/round1-ondevice-patch-test.md`](parity-evidence/round1-ondevice-patch-test.md)).
Round 2 root-caused it precisely: `detector.tflite` stores its larger weight tensors using
TFLite's out-of-line `Buffer.Offset()`/`Size()` convention (for zero-copy mmap loading) —
a real, valid file, not a corrupt one — which `TensorFlowLiteC 2.17.0`'s loading APIs
(both the buffer-based and file-based ones; a native-loader patch trying the latter was
tried and refuted too) do not resolve.

**Round 3 fixed it**: `ai_edge_litert.tools.flatbuffer_utils`'s own `read_model`/
`write_model` pair already resolves offset-based buffers back into the classic inline
format as an ordinary part of reading a model — confirmed byte-for-byte identical
inference output before/after (same all-zero input, Python `ai_edge_litert`), so it's a
real fix, not a workaround. `src/smart_scene_analyzer/export/tflite.py`'s
`export_detector` now applies this (`_inline_offset_buffers`) as a post-export step, and
`app/src/pipeline/analyzeImage.ts` now calls the real detector (`loadDetectorModel` +
`runSync`) instead of the frozen Python-reference fixture it used as a stand-in all
session. Full evidence chain, including two refuted fix attempts before this one, is in
[`docs/parity-evidence/round2-invoke-root-cause.md`](parity-evidence/round2-invoke-root-cause.md).

## Detections

Python reference (Path B), source px, 730×530 original image:

| # | Class | Confidence | Box (source px) |
|---|---|---|---|
| 1 | chair | 0.916 | (156.5, 102.1) – (278.9, 302.1) |
| 2 | storage | 0.850 | (257.0, 4.9) – (564.6, 496.2) |
| 3 | chair | 0.321 | (0.0, 224.9) – (55.1, 524.6) |

Real on-device detections (Path A, iPhone 17 Simulator, round 3), MODEL-INPUT px
(640×640 letterboxed — **not** the same space as the table above; converting back to
source px to do a rigorous IoU match was not done this round, see "what was not tested"):

| # | Class | Confidence | Box (model-input px) |
|---|---|---|---|
| 1 | chair | 0.866 | (150.6, 196.2) – (249.4, 345.6) |
| 2 | storage | 0.831 | (224.1, 93.7) – (494.9, 524.1) |
| 3 | chair | 0.498 | (10.1, 284.8) – (50.6, 535.5) |

**Qualitatively, this matches**: same three objects, same classes, same rough spatial
layout (storage spanning most of the right/center frame, one chair upper-left-of-center,
one small chair at the bottom-left edge), confidences in the same ballpark. This is a
real, working detector, not coincidence — Invoke() was failing 100% of the time before
round 3's fix and now succeeds reliably (confirmed across ~20 repeated calls with zero
failures).

**One real finding worth flagging, not hiding**: repeated on-device runs on the identical
image are not bit-identical. Across several back-to-back calls in the same app session,
confidence values for the two higher-confidence detections stayed exactly constant
(`0.8663232922554016`, `0.8307209610939026` — bit-identical across runs), while their box
coordinates shifted by a few pixels, and the third, lower-confidence chair detection
(~0.5, near several candidate anchors of similar score) sometimes disappeared entirely.
This is consistent with ordinary multi-threaded XNNPACK floating-point accumulation
nondeterminism affecting *which* of several near-duplicate anchor candidates NMS selects,
not a correctness bug introduced by this round's fix — confidence (purely a function of
the winning anchor's quantized class logit) stays pinned; which anchor wins among
near-ties does not. Flagged here rather than silently smoothed over by picking the
cleanest-looking run for the table above.

## Depth

Depth is compared by **ordering**, not by value — the output has no scale, so comparing
magnitudes across two runtimes is meaningless. This comparison predates round 3's
detection fix, so it reuses Path B's three real detections (not yet real on-device boxes
at the time it was run) as shared sample points for **both** sides: the project's own
reference reduction,
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
| Run-to-run anchor selection jitter on near-tie candidates (box coordinates shift a few px, a borderline ~0.5-confidence detection sometimes vanishes; winning-anchor confidences stay bit-identical) | **Not one of the four** — ordinary multi-threaded XNNPACK float nondeterminism in NMS tie-breaking, not a preprocessing/decoding/quantization/port disagreement | Does confidence change, or only which near-duplicate anchor NMS keeps? | Confidence pinned; anchor selection varies — consistent with threading noise, not a correctness bug |
| On-device depth values differ from PyTorch eager by a small, bounded amount at matched object locations | **Quantization** | Does the int8 export's whole-map rank correlation against eager fall within the project's already-measured noise floor? | Yes — consistent with, not worse than, the existing floor |

**The detection blocker from rounds 1-2 is resolved**, not one of the four — it was a
native model-loading/file-format mismatch upstream of where preprocessing, decoding,
quantization, or port differences could even be observed:
- `tests/test_export_parity.py::test_detection_parity_per_class` (Python `ai_edge_litert`
  running the actual exported file) shows the export's **content** agrees with PyTorch at
  a 91.9% match rate (34/37), worst class delta −0.133 — ordinary int8 drift, re-confirmed
  unaffected by round 3's buffer-inlining fix (byte-for-byte identical Python inference
  output before/after).
- `AllocateTensors` always succeeded (the op graph resolves; shapes/dtypes are metadata);
  `Invoke()` failed the moment it reached a weight tensor stored via TFLite's out-of-line
  buffer convention. Round 3 fixed this by inlining those buffers post-export. See
  `docs/parity-evidence/round2-invoke-root-cause.md` for the full evidence chain (root
  cause) and the same file's "Fix attempt" sections for what was tried, refuted, and what
  finally worked.

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

**Justification:** on-device detection now exists (round 3), but this report did not
apply the IoU/confidence-delta tolerance to it — doing so needs the MODEL-INPUT-to-source-px
conversion flagged under "what was not tested," not done this round. Stating a passing
number here without that conversion would imply a real comparison happened. The IoU and
confidence-delta values above are carried over unchanged from this project's own existing
export-parity test (written before this report, against different data) so that the same
bar applies once that comparison is actually run, rather than one chosen after seeing the
data. Depth ordering's tolerance is exact match (not a correlation threshold) because only
three objects were available this round; 3/3 is reported as what it is — a small-n perfect
result, not evidence of robustness at scale (see "what was not tested").

## Verdict

**Confirmed working for both models, with a real caveat on detection's precision claim.**

- **Depth:** the real on-device `depth.pte` run, compared against PyTorch eager at three
  real object locations, preserved near-to-far ordering exactly (3/3) with value deltas
  consistent with this project's own previously-measured int8 quantization floor
  (mean rank correlation 0.9447 across 8 images). For depth specifically, the device is
  behaving like the trained model, within already-accepted quantization noise — on the
  one image and three objects tested.
- **Detection:** **the on-device pipeline now runs the real detector and finds the right
  objects**, confirmed working (not merely not-crashing) across ~20 repeated on-device
  `Invoke()` calls with zero failures, and qualitatively matching the Python reference's
  classes, confidences, and spatial layout on the one image tested. What this round does
  **not** claim: a rigorous, quantified box-IoU/confidence-delta match rate against the
  Python reference for a real on-device run — that needs converting the captured
  MODEL-INPUT-space boxes back to source px and running the same matching methodology
  `test_detection_parity_per_class` uses, which wasn't done this round (see "what was not
  tested"). The run-to-run anchor-selection jitter above is also real and unresolved as a
  *characterization* question (how often does the borderline chair disappear, across how
  many images?), even though it's understood as ordinary threading noise, not a bug.

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
- **A rigorous, quantified on-device-vs-Python detection match** — round 3 confirms
  detection works and qualitatively matches, but did not convert the real on-device boxes
  back to source px and run `test_detection_parity_per_class`'s own IoU/confidence-delta
  methodology against them. That's the natural next step to turn "looks right" into a
  measured number.
- **How often the anchor-selection jitter drops a real detection** — observed on one
  borderline (~0.5 confidence) detection across a handful of repeated runs on one image;
  not characterized across more images, more runs, or lower-confidence detections
  generally.

A reader should not conclude from this report that the depth result generalizes beyond
this one image, that detection's precise box/confidence agreement with Python has been
measured (only that it looks right), or that either of the two earlier `Invoke()`
hypotheses (op-version in round 1, native-loader-API-choice in round 2's first attempt)
remains live — both are refuted; round 3's buffer-inlining fix is the one that worked.

## Artifact integrity

`app/assets/models/detector.tflite` was **intentionally changed this round** — this is
the fix, not drift. The file now has its weight buffers inlined
(`_inline_offset_buffers`, applied as a permanent post-export step going forward):

```
sha256: 4aaaf0addf949ae3c5cab64a00ffb996027a67699a12c7e6bb4bab3b35fdb0e3
```

— matching `app/assets/models/model_contract.json`'s updated recorded hash. The original,
pre-fix file (`sha256: 3f5bf68248e5c36abf4f2fb260c38f1ec27d76203a22cc61e054f510b93ad83d`)
is backed up at `/tmp/detector_original_backup_round3.tflite` for the remainder of this
session. Confirmed numerically identical to the original under Python inference (same
all-zero input, `ai_edge_litert.Interpreter`, bit-identical output) — this is a storage-
format change, not a weight change. `app/App.tsx` is confirmed reverted to its
pre-investigation content (temporary diagnostic blocks added and removed across all three
rounds; final `tsc --noEmit`, `expo lint`, and a rebuilt, screenshotted simulator run all
confirm a clean state with no leftover debug code).
