# Architecture — Smart Scene Analyzer

- **Status:** design, Lesson 01. No code behind it yet.
- **Date:** 2026-09-01
- **Held to:** [`docs/requirements.md`](requirements.md)
- **Constrained by:** [ADR 0001 — Inference target: on-device](decisions/0001-inference-target.md)

This document defines the components, the contract at every boundary between them, the
fusion strategy, and the allocation of the N1 latency budget. It does not implement
anything.

Four decisions are **open** and this design refuses to close any of them: the device
floor, the object taxonomy, MLflow hosting, and whether two native runtimes are affordable
at all. [§10](#10-open-decisions-this-design-does-not-close) states each branch and its
architectural consequence. The design is drawn so that three of the four change an
implementation behind a contract rather than the contract itself; the fourth
(one runtime or two) reaches the latency budget, and that is said where it bites.

---

## 1. Deployment boundaries

There are two artifacts in this system that ship on different timetables, and the seam
between them is the only thing they share.

```
╔═══ BOUNDARY A — the handset ═══════════════════════════════════════════════════╗
║                                                                                ║
║   app/   Expo development build · TypeScript · iOS + Android                   ║
║                                                                                ║
║   Ships through an app-store review. A model bug here is not fixed by          ║
║   editing the file next to it — it is fixed by an export, a rebuild, a         ║
║   submission, and a review queue. This is the cost ADR 0001 accepted.          ║
║                                                                                ║
║   No network call exists anywhere inside this boundary on the inference path.  ║
║   (N1a = 0 bytes · F10 airplane mode · N15 no telemetry on the path)           ║
╚════════════════════════════════════════════════════════════════════════════════╝
                              ▲
                              │   THE ARTIFACT CONTRACT  (C11)
                              │   two model files + one contract manifest
                              │   copied into app/assets/models/ at build time
                              │   — crossed at build time, never at run time —
                              ▼
╔═══ BOUNDARY B — the workstation and CI ════════════════════════════════════════╗
║                                                                                ║
║   src/smart_scene_analyzer/   Python                                           ║
║                                                                                ║
║   Trains, exports, evaluates, and remains the REFERENCE IMPLEMENTATION         ║
║   the device is checked against (N20 export parity, N21 fusion port parity).   ║
║   Ships on a `git push`. Never ships to a user.                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

The two boundaries have opposite release economics, and that asymmetry is the reason the
parity tests are quality gates rather than nice-to-haves. Boundary B can be fixed in
minutes; boundary A cannot. So everything that *can* be proven at boundary B is proven
there, on fixtures, before it is allowed to cross.

---

## 2. Component diagram — the on-device pipeline

Every box below runs inside Boundary A. Stage IDs `S0`–`S8` are referenced by the latency
budget in [§6](#6-latency-budget-n1--400-ms-p95). Contract IDs `C1`–`C13` are defined in
[§4](#4-data-contracts).

```
                    ┌───────────────────────────────┐
                    │  S0  Acquisition              │   expo-camera / expo-image-picker
                    │      capture or pick          │
                    │      + downscaling decode     │   ← decodes SUB-SAMPLED (N7)
                    └───────────────┬───────────────┘
                                    │  C2  SourceImage
                                    │      uint8 [H_src, W_src, 3] RGB, SOURCE space
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
   ┌──────────▼───────────┐                    ┌──────────▼───────────┐
   │ S1 Preprocess-D      │                    │ S2 Preprocess-Z      │
   │    letterbox + quant │                    │    letterbox + norm  │
   │    to DET_INPUT      │                    │    to DEPTH_INPUT    │
   └──────────┬───────────┘                    └──────────┬───────────┘
              │ C3 DetectionInput                         │ C4 DepthInput
              │    + C3t LetterboxTransform               │    + C4t LetterboxTransform
              │                                           │
   ┌──────────▼───────────┐                    ┌──────────▼───────────┐
   │ S3 DetectorBackend   │                    │ S4 DepthBackend      │
   │    YOLO11 int8       │◄── pluggable ──►   │    Depth Anything V2 │
   │    .tflite           │    (see §10.4)     │    .pte              │
   │    fast-tflite       │                    │    executorch        │
   └──────────┬───────────┘                    └──────────┬───────────┘
              │ C5 RawDetectorOutput                      │ C8 DepthMap
              │                                           │    DEPTH_INPUT space
   ┌──────────▼───────────┐                                │
   │ S5 Decode + NMS      │                                │
   │    (pure TypeScript) │                                │
   └──────────┬───────────┘                                │
              │ C6 list<Detection>, DET_INPUT space        │
   ┌──────────▼───────────┐                                │
   │    unletterbox       │  the one named conversion      │
   └──────────┬───────────┘                                │
              │ C7 list<Detection>, SOURCE space (F3)      │
              │                                            │
              └────────────────────┬───────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │  S6  SCENE UNDERSTANDING LAYER          │
              │      app/src/fusion/index.ts            │
              │      PORT of src/…/fusion.py  (N21)     │
              │      pure: arrays in, structure out     │
              │      tolerates depth_map = null (N12)   │
              └────────────────────┬────────────────────┘
                                   │  C9  SceneResult
                                   │      boxes SOURCE space
                                   │      + relative_depth per object
              ┌────────────────────▼────────────────────┐
              │  S7  Overlay render                     │
              │      toScreen(): SOURCE → SCREEN (C10)  │
              │      boxes · labels · depth shading     │
              │      empty list ≠ loading ≠ failed (F8) │
              └─────────────────────────────────────────┘

  S8  Orchestration — the analyse() driver, model lifecycle, and the JS↔native
      marshalling tax charged on every arrow that crosses into a runtime.
```

### Two corrections this diagram makes to the sketch in `CLAUDE.md`

1. **There is not one letterbox node. There are two.** The sketch shows a single
   `letterbox · normalize` fanning out to both models. The two models have different input
   resolutions and different input dtypes, so they need different letterbox transforms,
   and therefore **two model-input coordinate spaces, not one**. Collapsing them is exactly
   the mistake that produces boxes that are wrong by a scale factor and still in range.
   The design names them `DET_INPUT` and `DEPTH_INPUT` and never lets them meet.

2. **`app/` is a component, not a container.** It has its own release timetable
   (§1), its own lifecycle-owning module (`ModelHost`, §5.3), and it is where the depth
   number is *computed* — not merely displayed. That is why `units_guard.py` extends to it.

---

## 3. Coordinate spaces

Three spaces, as `CLAUDE.md` says — but `MODEL_INPUT` is two distinct instances, so in
practice there are four frames on the path. Every contract in §4 names its space.

| Space | Definition | Origin / axes | Units | Who produces it |
|---|---|---|---|---|
| `SOURCE` | The **decoded working image**. EXIF orientation already applied, so it is display-upright. Long edge ≤ `DECODE_LONG_EDGE` (see A1, A10). | top-left, x→right, y→down | absolute pixels | S0 |
| `DET_INPUT` | The letterboxed detection-model input grid, `DET_SIZE × DET_SIZE`. Has its own scale and pad offsets. | top-left of the padded canvas | absolute pixels | S1 |
| `DEPTH_INPUT` | The letterboxed depth-model input grid, `DEPTH_SIZE × DEPTH_SIZE`. **Different scale and pads from `DET_INPUT`.** | top-left of the padded canvas | absolute pixels | S2 |
| `SCREEN` | The image view on the handset, `resizeMode: contain`. | top-left of the **view**, not the image | density-independent points (React Native points, not physical pixels) | S7 |

**Conversion rule.** A conversion between two spaces happens in exactly one named function
whose signature names both spaces, and nowhere else:

```
letterbox_forward(box_xyxy_source, transform)  -> box_xyxy_model_input
letterbox_inverse(box_xyxy_model_input, transform) -> box_xyxy_source
to_screen(box_xyxy_source, viewport)           -> box_xyxy_screen
```

A `LetterboxTransform` is `{scale: float, pad_x: float, pad_y: float, in_w, in_h, out_w, out_h}`
and it is **not interchangeable between models**. Passing the detection transform to a
depth lookup returns a number, in range, and wrong.

> **⚠️ FLAG — requirements are silent.** F3 says the bbox is in "the **source image**
> frame". It does not say whether that means the pixel grid of the *file the picker
> returned* or the *decoded working image*. N7 forces a downscaling decode (a 3024×4032
> RGBA decode is ~48.8 MB resident before anything else runs), so those two are not the
> same grid. This design defines `SOURCE` as the decoded working image and requires
> `SourceImage` to carry `original_size` and `decode_scale` so the file's grid is
> recoverable. If F3 meant the file's grid, one multiplication moves it — but somebody has
> to say which, and requirements did not.

---

## 4. Data contracts

Every boundary. Type, units, coordinate space, and what a degenerate value means.

Fields marked **export-owned** are fixed by the export step and *assumed* by the app;
nothing checks them at run time (`CLAUDE.md`, "The artifact contract"). Their concrete
values below are **provisional until Lesson 04's export lands** and are then authoritative
from `model_contract.json` (C11), not from this document.

---

### C1 — Acquisition input → S0

```
ImageRef {
  uri:            string        // file:// or ph:// — a local handle. Never a URL.
  reported_size:  {w: int, h: int} | null   // what the picker claims, in FILE pixels
}
```
No bytes leave the device to obtain this. `uri` is a local handle; a remote scheme is a
defect, not a supported case.

---

### C2 — S0 → S1, S2, S7 · `SourceImage`

```
SourceImage {
  pixels:        uint8[H_src, W_src, 3]   // RGB, channel-last (HWC), not BGR, not RGBA
  size:          {w: W_src, h: H_src}     // SOURCE space, absolute pixels
  original_size: {w: int, h: int}         // the FILE's grid, before decode downscale
  decode_scale:  float                    // SOURCE px per FILE px, ≤ 1.0
}
```
- **Space:** `SOURCE`. **Units:** absolute pixels. **Range:** `[0, 255]` per channel.
- EXIF orientation is applied during decode. Downstream code never re-rotates.
- Alpha is dropped at decode; there is no alpha anywhere after S0.

---

### C3 — S1 → S3 · `DetectionInput` (**export-owned**)

```
DetectionInput {
  tensor:    <dtype>[1, DET_SIZE, DET_SIZE, 3]   // NHWC — layout is part of the contract
  transform: LetterboxTransform                   // SOURCE → DET_INPUT
}
```
- **Space:** `DET_INPUT`. **Units:** absolute pixels of the padded canvas.
- **Provisional values (A2, A4):** `DET_SIZE = 640`; pad value `(114, 114, 114)`;
  aspect-preserving letterbox, padding split evenly on both sides.
- `dtype`, the quantization `scale`/`zero_point`, and whether values are `[0,1]`-scaled
  before quantization are **fixed by the int8 export** and read from C11. Writing them
  from memory is the failure mode this contract exists to prevent.
- `[1, 640, 640, 3]` uint8 NHWC is **not** interchangeable with `[1, 3, 640, 640]` float32
  NCHW. A silent swap produces plausible garbage.

**C3t — `LetterboxTransform`** (accompanies C3 and C4, never travels alone):
```
LetterboxTransform {
  scale: float    // model-input px per SOURCE px
  pad_x: float    // model-input px added on the LEFT
  pad_y: float    // model-input px added on the TOP
  in_w, in_h:  int   // SOURCE size it was computed from
  out_w, out_h: int  // model-input canvas size
}
```
Forward: `x_model = x_source * scale + pad_x`. Inverse: `x_source = (x_model - pad_x) / scale`.

---

### C4 — S2 → S4 · `DepthInput` (**export-owned**)

```
DepthInput {
  tensor:    float32[1, 3, DEPTH_SIZE, DEPTH_SIZE]   // NCHW — note: differs from C3
  transform: LetterboxTransform                       // SOURCE → DEPTH_INPUT
}
```
- **Space:** `DEPTH_INPUT`. **Provisional values (A3):** `DEPTH_SIZE = 518`, ImageNet
  channel normalization, NCHW float32.
- **The layout differs from C3 on purpose**, because the two runtimes differ. Any code
  that builds both tensors in one function must state both layouts in its signature.
- `transform` here is **not** C3's transform. Different scale, different pads.

---

### C5 — S3 → S5 · `RawDetectorOutput` (**export-owned**)

```
RawDetectorOutput {
  tensors: <dtype>[...]    // count, order, shape, and dtype fixed by the export
}
```
- **Space:** whatever the export emits, normalized or `DET_INPUT` pixels — **stated in
  C11, not assumed here.** Ultralytics emits `cxcywh` normalized to `[0,1]` in some export
  paths and `DET_INPUT` pixels in others; guessing costs an afternoon.
- **Contract requirement:** C11 must state, per output tensor, its index, shape, dtype,
  box encoding (`cxcywh` / `xyxy`), and whether coordinates are normalized. An export that
  does not emit this is not a finished export.

---

### C6 — S5 (decode+NMS) → unletterbox · `list<Detection>` in `DET_INPUT`

```
Detection {
  bbox:            [x1, y1, x2, y2]   // xyxy, ABSOLUTE PIXELS, DET_INPUT space
  class_id:        int                // index into the label list in C11
  label:           string             // resolved from C11's ordered labels
  confidence:      float              // [0.0, 1.0]
  relative_depth:  null               // not yet computed at this boundary
}
```
- `x1 ≤ x2`, `y1 ≤ y2`, invariant, enforced at construction.
- Boxes here may overlap the letterbox padding bands; they are **not** clipped yet.
- **Empty list is a valid, successful value** (F8), at this boundary and every one below.

---

### C7 — unletterbox → S6 · `list<Detection>` in `SOURCE` **(the boundary that loses afternoons)**

```
list<Detection>   // identical shape to C6, one field changed in meaning:
                  // bbox is xyxy in ABSOLUTE PIXELS of the SOURCE image frame —
                  // NOT the letterboxed DET_INPUT frame.
```
- Produced by `letterbox_inverse(box, det_transform)`, then **clipped to
  `[0, W_src] × [0, H_src]`**. Clipping happens here and only here.
- A box that clips to zero area is **dropped**, not kept with a degenerate box. It was
  entirely inside the padding band, which means it was never in the image.
- This satisfies F3. Anything downstream that receives a `Detection` may assume `SOURCE`.

---

### C8 — S4 → S6 · `DepthMap`

```
DepthMap {
  values:    float32[DEPTH_SIZE, DEPTH_SIZE]   // 2-D, no batch, no channel
  transform: LetterboxTransform                // SOURCE → DEPTH_INPUT (C4's, reused)
  valid:     {x1, y1, x2, y2}                  // the un-padded content rect, DEPTH_INPUT px
}
```
- **Space:** `DEPTH_INPUT`, absolute pixels. **It is not resampled to `SOURCE`** — see
  R4 for why.
- **Units: none.** The values are **relative inverse depth**: larger is nearer, the scale
  is arbitrary, and it is comparable **only within this one image**. There is no
  conversion to a physical distance and the system may not state one (out of scope,
  `units_guard.py`).
- Values outside `valid` are letterbox padding. **Padding is never sampled** — it is grey
  canvas the model hallucinated a depth for, and it is systematically far.
- `NaN`/`Inf` in `values` is a model defect; S6 treats such samples as invalid (§5.2).

---

### C9 — S6 → S7 · `SceneResult`

```
SceneResult {
  detections: list<FusedDetection>
  depth_available:   bool          // false ⇒ the depth branch failed (N12)
  source_size:       {w, h}        // SOURCE space, for the S7 conversion
  detector_version:  string        // F7 — from C11
  depth_version:     string | null // F7 — from C11, null when depth_available = false
  depth_preview:     DepthPreview | null   // F9, see below
}

FusedDetection extends Detection {
  bbox:                       [x1,y1,x2,y2]   // xyxy, absolute px, SOURCE space
  relative_depth:             float | null    // raw median of the sampled region.
                                              // Unit-less relative inverse depth,
                                              // larger is nearer. null ⇒ not derivable.
  relative_depth_normalized:  float | null    // [0,1] within THIS image only. 1 = nearest.
  depth_rank:                 int | null      // 0 = nearest in this image. Ties broken by
                                              // descending relative_depth then class_id.
  depth_spread:               float | null    // [0,1]. Robust IQR of the sampled region
                                              // ÷ the image's robust depth range.
                                              // High ⇒ the box straddles a depth edge.
  depth_sample_count:         int             // how many valid samples produced the above
}

DepthPreview {
  shade:  uint8[DEPTH_SIZE, DEPTH_SIZE]   // 0–255, for shading ONLY. Not a measurement.
  transform: LetterboxTransform
}
```
- `relative_depth = null` is the **defined** result when depth is unavailable (N12) or the
  region yielded no valid samples. It is not an error and not `NaN`. `NaN` must not appear
  in this structure at all.
- `detections = []` with `depth_available = true` is a **successful empty result** (F8).
- Nothing in this structure is a distance, and no field name may imply one.

---

### C10 — S7 · `SOURCE → SCREEN`

```
to_screen(bbox_source, viewport) -> bbox_screen
viewport {
  view_w, view_h:  float   // SCREEN points
  image_w, image_h: int    // SOURCE pixels
  fit: "contain"           // A9
}
```
- **Units in:** SOURCE absolute pixels. **Units out:** density-independent SCREEN points.
- Contain-fit introduces a *third* scale and a *third* pair of offsets. It is not the
  detection letterbox and not the depth letterbox, and it changes on rotation and on
  split-view. It is recomputed per layout pass, never cached across one.

---

### C11 — Boundary B → Boundary A · **the artifact contract**

Three files, copied into `app/assets/models/` at build time. The `.tflite` and `.pte` are
gitignored; the manifest is not, because it is the only part a reviewer can read.

```
model_contract.json {
  detector: {
    file, sha256, registry_version,      // F7, N19 — an explicit registry version
    input:  { shape, dtype, layout, quant: {scale, zero_point} | null,
              normalization, letterbox_pad_value },
    outputs:[{ index, name, shape, dtype, encoding: "cxcywh"|"xyxy",
               normalized: bool }],
    labels: [ ... ],                     // ORDERED. Index = class_id. Taxonomy: OPEN (§10.2)
    default_conf_threshold, default_iou_threshold, max_detections
  },
  depth: {
    file, sha256, registry_version,
    input: { shape, dtype, layout, normalization },
    output:{ shape, dtype, semantics: "relative-inverse-depth-larger-is-nearer" }
  },
  exported_at, exporter_version, dataset_version
}
```
- **Label order is the highest-risk field in this document.** An app built against last
  week's order runs, draws boxes, and names them wrong, and no test on either side of the
  boundary catches it. `sha256` + `artifact_drift.py` is the mitigation.
- `semantics` is a literal string, checked by the app at load, so that a future export that
  changes depth polarity fails loudly instead of rendering the scene inside-out.

---

### C12 — Reference ↔ port · **the fixture contract** (N21)

```
tests/fixtures/fusion/<case>.json {
  depth_map:   { size: [h, w], values: [ ... ] },   // row-major, DEPTH_INPUT space
  transform:   LetterboxTransform,
  detections:  [ { bbox: [x1,y1,x2,y2], class_id, confidence } ],  // SOURCE space
  source_size: {w, h},
  expected:    { detections: [ { relative_depth, relative_depth_normalized,
                                 depth_rank, depth_spread, depth_sample_count } ] },
  tolerance:   { relative_depth: 1e-5, normalized: 1e-5 }
}
```
- **Plain JSON. No pickles, no `.npy`.** A fixture the TypeScript suite cannot load tests
  half the system.
- Row-major, `float` as JSON numbers. Both sides parse to float64 and compare within
  `tolerance`.
- When the two disagree, **that is the finding.** The fixture is not adjusted.

---

### C13 — Training → tracking (**hosting OPEN, §10.3**)

```
RunRecord {
  seed, dataset_version, config, metrics, artifacts   // N19
}
```
Written through a `RunTracker` protocol. The tracking URI comes from
`MLFLOW_TRACKING_URI`; **no module hardcodes a backend or a hosting arrangement**, because
that decision is open. `NullTracker` is the offline test double.

---

## 5. The Scene Understanding Layer (S6)

`src/smart_scene_analyzer/fusion.py` is the reference. `app/src/fusion/index.ts` is the
port. They are held together by C12 and by nothing else.

### 5.1 The reduction: median over an inset box, sampled on a bounded grid

Given one `Detection` (C7, `SOURCE` space) and one `DepthMap` (C8, `DEPTH_INPUT` space):

1. **Map the box into the depth grid.** `letterbox_forward(bbox_source, depth_transform)`.
   The depth transform, never the detection transform. This is the only conversion in the
   layer.
2. **Intersect with `valid`.** Clip to the un-padded content rect so no padding pixel is
   ever read.
3. **Inset the box.** Shrink by `INSET_FRACTION` of each side (A5: `0.125`, i.e. keep the
   central 75% × 75%), floor-clamped so each side keeps at least 1 px. The corners of a
   box are where background lives; the centre is where the object lives.
4. **Sample on a bounded grid.** Stride the inset region so at most `MAX_SAMPLES` points
   are read (A6: 1024). Cost becomes independent of box area, which is what makes S6's
   latency allocation a number rather than a hope.
5. **Drop invalid samples** — `NaN`, `Inf`.
6. **`relative_depth` = the median** of the surviving samples.
7. **`depth_spread`** = `IQR(samples) / robust_range(whole depth map)`, clamped to `[0,1]`,
   where `robust_range` is the 2nd-to-98th percentile span of `values` inside `valid`.
8. **`relative_depth_normalized`** = `relative_depth` positioned in that same robust range,
   clamped to `[0,1]`. Percentiles, not min/max: one hot pixel must not compress every
   object in the image into the bottom of the scale.
9. **`depth_rank`** = index after sorting descending by `relative_depth` (0 = nearest).

### 5.2 Degenerate cases — every one has a defined result

| Case | Result |
|---|---|
| Box has zero area in `SOURCE` | Dropped at C7; never reaches S6 |
| Inset collapses to < 1 px | Fall back to the un-inset clipped box |
| Box lies entirely in the padding band | `relative_depth = null`, `depth_sample_count = 0`. Not an error, not `NaN` |
| All samples `NaN`/`Inf` | Same as above |
| `depth_map = null` (N12, depth failed to load) | Every `relative_depth = null`, `depth_available = false`, ranks `null`. Detections still render |
| Zero detections | `SceneResult` with `detections: []`, `depth_available` unchanged (F8) |
| All objects at the same depth (`robust_range ≈ 0`) | `relative_depth_normalized = 0.5` for all; ranks by class_id tiebreak. No division by zero |

**`NaN` never leaves this layer.** It becomes `null`, which the UI can render as
"no depth for this object" — a state, not a silent zero that draws as "nearest".

### 5.3 What a background gap does to it

A background gap inside the box is the normal case, not the exception — a chair's box
contains the floor between its legs, and every non-convex object does something similar.

- **The median tolerates it up to 50% contamination by construction.** That is the whole
  argument for the median over the mean: the mean moves with every background pixel;
  the median does not move at all until background is the majority.
- **The inset removes the cheapest background** — the corners, which are background in
  almost every box, including the convex ones.
- **Above 50% contamination the median follows the background, and it does so silently.**
  A thin, elongated, or heavily-occluded object — a lamp stand, a chair back — can be a
  minority of its own bounding box. This design does not pretend to fix that. It
  **reports** it: `depth_spread` is high exactly when the sampled region straddles a depth
  discontinuity, which is the signature of a box that is mostly background. The UI may
  de-emphasize a high-spread box; evaluation (N4) may stratify ordering accuracy by it.
  A number the system knows is unreliable and does not say so is worse than no number.

The honest fix is a segmentation mask, and **instance segmentation is out of scope**
(`docs/requirements.md`, "Out of scope"). So the reduction is a box statistic, and the
spread is how it admits what it does not know.

### 5.4 Rejected reductions

| Method | Why not |
|---|---|
| Mean over the box | Every background pixel moves it. The corners alone bias it far |
| Max (nearest sample) | One noisy pixel decides the object's depth. Least robust of all |
| Centre pixel | Zero cost, fails on any object whose centre is a gap — chairs, tables, bicycles |
| Masked region | Correct, and unavailable: it needs segmentation, which is out of scope |
| Median over the **full** box | The default fallback, and it is what step 3 degrades to. Strictly worse than the inset on convex objects and no better on the rest |

---

## 6. Latency budget — N1 ≤ 400 ms p95

**Scope.** On-device, release build, single 1280×720 image, capture/pick to drawn overlay.
**Model load is excluded** — that is N6, a separate budget in §7, and folding it in here
would hide a 3-second cost behind a 400 ms number.

**There is no transit line.** ADR 0001 deleted the term. Every millisecond below is
compute the device performs, which is why the reference device matters as much as a link
speed used to.

**These are allocations, not measurements.** Nobody has run this. Each is a ceiling a
stage must be shown to fit under, and the first measurement that exceeds one is a finding
about that stage, not a reason to re-cut the table.

**Reference device.** The N1 measurement reference — iPhone 13 (A15) and Pixel 6a
(Tensor G1). **Not the device floor, which is open (§10.1).** If the floor lands below
these, this entire table is re-derived and N1 may not survive.

### 6.1 The allocation (concurrent inference)

| ID | Stage | Allocation | Note |
|---|---|---|---|
| S0 | Acquire + downscaling decode → `SourceImage` | **35 ms** | Sub-sampled decode, not full decode + resize (N7) |
| S1 | Preprocess-D: letterbox + quantize → `DetectionInput` | **20 ms** | 640×640, native |
| S2 | Preprocess-Z: letterbox + normalize → `DepthInput` | **25 ms** | 518×518 float32 NCHW; more bytes than S1 |
| S3+S4 | **Inference fan-out**, concurrent | **185 ms** | `max(S3 = 90, S4 = 150)` **+ 35 ms contention tax** |
| S5 | Decode + NMS | **20 ms** | Bounded by `max_detections` (A7) |
| S6 | Fusion | **15 ms** | Bounded by `max_detections × MAX_SAMPLES` (A6, A7) |
| S7 | Overlay render to first drawn frame | **30 ms** | Two frames at 60 Hz |
| S8 | Orchestration + JS↔native marshalling | **15 ms** | Charged once, not per arrow |
| — | **Sum of allocations** | **345 ms** | |
| — | **Reserve** | **55 ms** | 13.75%. GC, thermal drift, layout thrash |
| — | **Total** | **400 ms** | **= N1** ✓ |

Arithmetic: `35 + 20 + 25 + 185 + 20 + 15 + 30 + 15 = 345`; `345 + 55 = 400`.

### 6.2 The serial arrangement, and why concurrency is load-bearing

If S3 and S4 run one after the other:

`35 + 20 + 25 + 90 + 150 + 20 + 15 + 30 + 15 = 400 ms`

**Exactly 400, with zero reserve.** A p95 target met with no margin is not met — p95 is a
tail statistic, and a budget with no slack fails the first time a GC pause lands in it.

So: **running the two models concurrently is not an optimization in this design, it is a
requirement.** The 55 ms reserve in §6.1 *is* the 55 ms that concurrency buys
(`240 − 185`). If the two runtimes cannot execute concurrently — because they contend on
one core cluster, because the bridge serializes them, or because §10.4 consolidates them
into a single runtime with a single inference lock — **N1 is not achievable as budgeted**
and one of the following must give:

- `DEPTH_SIZE` drops (256 instead of 518), trading N4 ordering accuracy for latency;
- detection drops to a smaller YOLO11 variant, trading N3 mAP for latency;
- N1 moves.

That is a decision for whoever measures it, not for this document. **What this document
states is the dependency**, so the measurement is taken knowing what it decides.

### 6.3 N2 (p50 ≤ 250 ms) cross-check

Applying a p50/p95 ratio of 0.70 per stage (A11) to the §6.1 allocations:
`345 × 0.70 = 241.5 ms` ≤ 250 ms, with **8.5 ms of margin**.

N2 is therefore the *tighter* constraint on the variance-heavy stages — S0 (image decode),
S7 (layout and GC), S8 (bridge) — because those carry nearly all of the p95-to-p50 spread.
The inference stages are close to deterministic. If the measured p50/p95 ratio proves
worse than 0.70, N2 fails before N1 does.

### 6.4 What is *not* in this budget

| Excluded | Where it lives |
|---|---|
| Cold model load | N6, §7 |
| The permission prompt, and the user's time in the picker | Not machine time |
| Network transit | **Does not exist** (ADR 0001, N1a = 0) |
| Telemetry | Forbidden on this path (N15) |
| First-run warmup inference | N6, §7 |

---

## 7. Cold model load budget — N6 ≤ 3.0 s (separate)

This is a **different budget**, paid once per app launch, with the user watching a spinner.
It is not part of N1 and must never be averaged into it.

| Step | Allocation | Note |
|---|---|---|
| TFLite interpreter creation + weight mapping | **400 ms** | int8, memory-mapped |
| ExecuTorch module load + `.pte` deserialization | **1400 ms** | The ViT is the expensive one |
| Delegate / backend initialization, both runtimes | **500 ms** | Core ML / NNAPI when present (N10: never a correctness dependency) |
| Warm-up inference, both models, on a dummy tensor | **600 ms** | Pays the first-call cost *here* so it is not charged to the user's first real tap and therefore not to N1 |
| Reserve | **100 ms** | |
| **Total** | **3000 ms** | **= N6** ✓ |

`400 + 1400 + 500 + 600 + 100 = 3000`.

**Design consequence.** Because N1 excludes model load and N6 is 3.0 s, the app must load
and warm both models at mount — not lazily on the first analysis. A lazy load puts N6's
3 seconds inside a user's first tap, where N1 is measured. `ModelHost` (§8) owns that
lifecycle, and the loading state it exposes is one of the three states F8 requires the UI
to distinguish.

---

## 8. Module boundaries, designed to be mockable

**The gate: `uv run pytest -m "not integration"` runs with no GPU, no network, and no
weights on disk.** Anything that cannot is behind a protocol or carries the marker.

Four seams do the work:

### 8.1 `inference/protocols.py` — the model seam

`DetectorBackend` and `DepthBackend` are `Protocol`s over arrays: tensor in, tensor out.
No file paths, no sessions, no device handles in the signature.

- The real implementations (`inference/detector.py`, `inference/depth.py`) import
  `ultralytics` / `transformers`, are constructed from a checkpoint, and are **only** used
  in `integration`-marked tests.
- A test double is a class with one method returning a canned `np.ndarray` loaded from
  C12's JSON. Structural typing means it does not import anything from this package.
- **This is also the seam §10.4 turns.** Consolidating onto one runtime replaces the
  implementations behind these two protocols and changes no contract in §4 except C11.

### 8.2 `fusion.py` — pure by construction

Arrays and dataclasses in, a dataclass out. No file reads, no model loads, no clock, no
randomness. It is testable with a hand-written 8×8 depth grid, and it is portable to
TypeScript *because* it is pure — an impure fusion layer could not be ported at all.

### 8.3 `geometry/` — no dependencies at all

`letterbox`, `spaces`, `boxes`. Pure arithmetic over numbers and small arrays. These are
the functions the coordinate-space bugs live in, so they are the ones with the cheapest
possible tests: a box, a transform, an expected box.

### 8.4 `training/tracking.py` — the tracking seam

`RunTracker` protocol; `NullTracker` for tests. Nothing under `src/` calls MLflow
directly, so no unit test needs a tracking server — which also means §10.3 stays open
without blocking anything.

### 8.5 Where the marker goes instead

| Module | Why it cannot be pure | Handling |
|---|---|---|
| `inference/detector.py`, `inference/depth.py` | Loads weights | `@pytest.mark.integration` |
| `export/tflite.py`, `export/executorch.py` | Produces artifacts | `@pytest.mark.integration` |
| `export/parity.py` | Compares artifact to reference (N20) | `@pytest.mark.integration` |
| `training/*` | GPU, dataset | `@pytest.mark.integration` |
| `preprocess/image_io.py` | Reads files | Tested on an in-memory buffer, not a path |

---

## 9. Rationale for each major choice

| # | Choice | Why |
|---|---|---|
| R1 | **Two letterbox transforms, never one** | The two models have different input sizes and dtypes. One shared transform is arithmetically wrong for at least one of them, and wrong in a way that returns an in-range number instead of an error |
| R2 | **Detections converted to `SOURCE` before fusion (C7)** | F3 requires it, and it means everything downstream of C7 — fusion, overlay, evaluation, fixtures — reasons in exactly one space. One conversion, in one named function, tested by fixtures |
| R3 | **Clip at C7 and nowhere else** | Clipping in two places means two sets of edge-case behaviour that agree until they do not |
| R4 | **The depth map is NOT resampled to `SOURCE`** | A 518×518 → 1280×720 float32 resample is ~0.9 M interpolations on the hot path, for no accuracy gain: fusion reads a few hundred samples out of it. Instead the box travels into the depth grid. The cost is that fusion must hold two transforms and be careful — which is why fusion is the module with the shared fixture set |
| R5 | **Median over an inset box** | Robust to background up to 50% contamination; the inset removes the corners, which are background in almost every box. See §5.4 for what was rejected and why |
| R6 | **`depth_spread` reported alongside `relative_depth`** | The median fails silently above 50% background. A system that reports a number it knows is unreliable, without saying so, is worse than one that says so |
| R7 | **Robust percentiles (2/98) for normalization, not min/max** | One hot pixel in a depth map would otherwise compress every object in the image into the bottom of the scale |
| R8 | **`null`, never `NaN`, for missing depth** | `NaN` propagates into a rendered overlay and draws as *something*. `null` is a state the UI must handle, and TypeScript will make it |
| R9 | **Models loaded and warmed at mount, not lazily** | N6's 3 s would otherwise be charged inside N1's 400 ms on the first tap (§7) |
| R10 | **Downscaling decode at S0** | A 3024×4032 RGBA decode is ~48.8 MB resident before either model loads. N7 says "handled without an out-of-memory failure"; decoding full-size and then resizing is the standard way to fail it |
| R11 | **`model_contract.json` beside the artifacts** | The artifact contract is currently enforced by nobody at run time. A manifest at least makes label order, tensor layout, and registry version *reviewable*, and gives the app one string (`semantics`) it can assert on at load |
| R12 | **Backends behind protocols** | It is what makes the offline suite possible (§8.1), and it is the seam the one-runtime fallback turns without touching any other contract |
| R13 | **Fixtures as plain JSON** | A fixture the TypeScript suite cannot load tests half the system. N21 is the only thing keeping two implementations of one algorithm honest |
| R14 | **Fusion tolerates a missing depth branch as a first-class input** | N12 is written as a design constraint, not error handling. `depth_map: DepthMap \| None` in the signature makes the degraded path a type the compiler checks, not a branch someone remembers |

---

## 10. Open decisions this design does not close

Each of these is genuinely open in `docs/requirements.md` or `docs/artifact-budget.md`.
This design is drawn to be invariant under three of them and states plainly where the
fourth reaches it.

### 10.1 Device floor — **open**

| Branch | Architectural consequence |
|---|---|
| Floor = the runtimes' own floor (iOS 17+ / Android 13+) | §6's budget is measured on hardware potentially much weaker than the iPhone 13 / Pixel 6a reference. The whole of §6 is re-derived, and N1 may not survive |
| Floor ≥ the N1 reference devices | §6 stands as written, and the supported device set shrinks — a product decision, not a technical one |

**Not resolved here.** §6 is explicitly allocated against the *measurement* reference and
labelled re-derivable. Nothing in §4 or §5 depends on it.

### 10.2 Object taxonomy — **open (Lesson 02)**

No class list appears anywhere in this design. `class_id` is an index into C11's ordered
`labels` array of length K; K is unknown and nothing computes on it. N3's mAP target stays
provisional.

**Not resolved here.**

### 10.3 MLflow hosting — **open**

| Branch | Consequence |
|---|---|
| Local file store | Zero setup; runs are not shareable; N19 satisfied for one person |
| Self-hosted server | A service to run, and one that must never appear on the inference path |
| Managed | An account and a bill outside the 20-credit ledger |

**Not resolved here.** C13 goes through a `RunTracker` protocol and reads
`MLFLOW_TRACKING_URI` from the environment. No module names a backend.

### 10.4 One runtime or two — **open, and this is the one that reaches the design**

| Branch | Consequence |
|---|---|
| **Two** (as sketched) | Two sets of native libraries against N9's 150 MB and N9a's 40 MB; two allocators against N8's 1.2 GB; **and the concurrency §6.2 depends on is at least plausible**, since the two runtimes have independent thread pools |
| **Consolidate on `.pte`** (YOLO11 → ExecuTorch) | One runtime, one set of libraries. The detection int8 path is re-validated end to end; N20 parity is re-run against a new artifact. **Both models likely share one inference queue — §6.2's 55 ms reserve may vanish and N1 becomes unachievable as budgeted** |
| **Consolidate on TFLite** (Depth Anything V2 → TFLite) | Same size win. Quantizing a ViT to int8 without unacceptable N4 loss is the unverified part. Same shared-queue risk to §6.2 |

**Not resolved here, and the design is arranged so that it does not have to be:** §4's
contracts C3, C4, C5, C8 and §8.1's protocols are unchanged by any of the three. What is
*not* invariant is the latency budget, and §6.2 states that dependency in full rather than
picking a branch to make the arithmetic work.

> **Nobody has measured the combined native library size.** Per
> `docs/artifact-budget.md`, this design does not treat the two-runtime split as final.

---

## 11. Assumptions

Everything below is a choice this design had to make that no requirement, ADR, or ledger
made for it. Each is falsifiable, and each is cheap to change *now* and expensive later.

| # | Assumption | If it is wrong |
|---|---|---|
| A1 | `DECODE_LONG_EDGE` = 1280 px for the downscaling decode | Changes `SOURCE` for every contract; changes N7's memory story |
| A2 | `DET_SIZE` = 640 (Ultralytics default) | C3 and every detection latency figure move |
| A3 | `DEPTH_SIZE` = 518 (Depth Anything V2 small default) | C4, C8, S4's 150 ms, and N4 all move |
| A4 | Letterbox pad value `(114,114,114)`, padding split evenly on both sides | Silent accuracy loss; the transform arithmetic changes if padding is one-sided |
| A5 | `INSET_FRACTION` = 0.125 per side (central 75%×75%) | Tunes §5.1. Larger insets are more robust on convex objects, worse on thin ones |
| A6 | `MAX_SAMPLES` = 1024 per box | S6's 15 ms allocation depends on it |
| A7 | `max_detections` = 100 per image | Bounds S5 and S6. Requirements are silent |
| A8 | Confidence threshold 0.25, NMS IoU 0.45 | **Requirements are silent and these move N3 directly.** They belong in C11, not in code |
| A9 | The result view uses contain-fit; `SCREEN` is React Native points, not physical pixels | C10's arithmetic |
| A10 | EXIF orientation is applied at decode, so `SOURCE` is display-upright and nothing re-rotates | Boxes appear rotated 90° on exactly the photos taken in portrait |
| A11 | p50/p95 ratio of 0.70 per stage, for the N2 cross-check (§6.3) | N2 fails before N1 if the real ratio is worse |
| A12 | The two runtimes can execute concurrently on separate native threads | **§6.2 — the 55 ms reserve, and therefore N1, depends on this.** Unverified |
| A13 | S4 (depth) is the longer inference of the two | Only changes which term is the critical path in §6.1; the total is unaffected |
| A14 | The depth map is produced on **every** analysis, including when there are zero detections | F5 says the app produces a dense depth map, so its cost is unavoidable and is budgeted unconditionally. If depth were skippable on an empty detection set, §6.1 would have a fast path it currently does not |
| A15 | Decode + NMS run in TypeScript on the device, per `CLAUDE.md`'s sketch, and are mirrored in Python as reference only | If NMS is instead baked into the exported graph, C5 changes shape and S5 disappears from §6.1 |
| A16 | F7's "model version" is the MLflow registry version, baked into C11 at export time | F7 otherwise has no on-device source, since there is no service to ask |
| A17 | The 1280×720 image in N1 arrives already at that size, i.e. S0's decode is a no-op scale for the N1 measurement | A 12 MP capture downscaled to 1280×720 costs more than S0's 35 ms allows |

---

## 12. Requirements are silent — flagged, not decided

| Question the design was forced to answer | Where |
|---|---|
| Does F3's "source image frame" mean the file's pixel grid or the decoded working image? | §3, resolved as the decoded working image **with `decode_scale` carried** so it is reversible |
| What are the confidence and NMS thresholds? | A8. They move N3 and they are not in requirements |
| What is the maximum number of detections per image? | A7. It bounds two latency allocations |
| Is the depth map computed when there are zero detections? | A14. F5 implies yes; nothing says so |
| Is `depth_spread` (or any reliability signal) part of the result? | §5.1. Requirements ask for a depth score and say nothing about confidence in it. This design emits one because §5.3's failure mode is otherwise silent |
| What does the UI do with a `null` `relative_depth`? | C9/§5.2 define the value. **The visual treatment is undefined** — and F8 only covers the zero-detections case, not the per-object missing-depth case |
| Where does the p50/p95 ratio in N2 come from? | A11. N1 and N2 are stated independently with no distributional assumption connecting them |

---

## 13. Directory scaffolding

`src/smart_scene_analyzer/`, created alongside this document. Module docstrings only —
no implementations.

```
src/smart_scene_analyzer/
├── __init__.py
├── types.py                   THE contracts of §4, as declarations
├── config.py                  dev-time settings from the environment
├── geometry/
│   ├── __init__.py
│   ├── spaces.py              the named coordinate spaces (§3)
│   ├── letterbox.py           SOURCE ↔ model-input transforms
│   └── boxes.py               xyxy operations: clip, inset, area, iou
├── preprocess/
│   ├── __init__.py
│   ├── image_io.py            bytes → SourceImage (C2)
│   ├── detection_input.py     SourceImage → DetectionInput (C3)
│   └── depth_input.py         SourceImage → DepthInput (C4)
├── inference/
│   ├── __init__.py
│   ├── protocols.py           the mock seam (§8.1)
│   ├── detector.py            reference YOLO11 backend
│   ├── depth.py               reference Depth Anything V2 backend
│   ├── decode.py              RawDetectorOutput → detections (C5→C6)
│   └── nms.py                 non-maximum suppression, pure
├── fusion.py                  THE reference scene understanding layer (§5)
├── scene.py                   assembles SceneResult; owns the N12 degraded path
├── export/
│   ├── __init__.py
│   ├── contract.py            writes and validates model_contract.json (C11)
│   ├── tflite.py              YOLO11 → int8 .tflite
│   ├── executorch.py          Depth Anything V2 → .pte
│   └── parity.py              N20: artifact vs. reference on fixtures
├── training/
│   ├── __init__.py
│   ├── dataset.py             dataset version → training inputs
│   ├── train_detector.py      the YOLO11 fine-tuning entry point
│   └── tracking.py            RunTracker protocol (C13, hosting OPEN)
├── evaluation/
│   ├── __init__.py
│   ├── detection_metrics.py   N3: mAP@50
│   ├── depth_ordering.py      N4: pairwise ordering accuracy
│   └── report.py              model-card inputs
└── fixtures/
    ├── __init__.py
    └── loader.py              C12 JSON fixtures, shared with the TS suite
```

Mirrored under `tests/` one file per module (`CLAUDE.md`: every module in `src/` has a
matching test file).

---

## 14. What is deliberately absent

| Absent | Why |
|---|---|
| An HTTP layer, a router, a request/response schema | ADR 0001. There is no service |
| A batching path | F6 is Won't. Batching is a server affordance with no user here |
| Any conversion from the depth score to a physical distance | Out of scope, and structurally prevented (`units_guard.py`) |
| Camera intrinsics, stereo, temporal tracking | Out of scope |
| A telemetry client | N15. Not even non-blocking — that would put a radio on a path this architecture guarantees is offline |
| A credentials store in `app/` | The app has nothing to authenticate to |
