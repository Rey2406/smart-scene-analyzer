# Live validation — one frame, qualitative

**This is a qualitative check on a single frame. It reports no accuracy figure, and none
should be read into it.** One frame is not a rate.

## Capture conditions

| Field | Value |
|---|---|
| Date | 2026-10-07 |
| Source | **File, not a webcam** — no camera was available in this environment |
| Frame | `data/v1/test/images/kv2_kinect2data_001187_2014-06-17_15-53-48_260595134347_rgbf000063-resize_0000063.jpg` |
| Resolution | 730 × 530 RGB |
| Scene | Indoor atrium: a large round table with four wooden chairs, a structural pillar, a tiled floor, a corridor receding to the right with two further chairs and a doorway, a blue bin beside the pillar |
| Lighting | Bright, even, daylight through the corridor; no strong shadow on the subjects |
| Detection weights | `yolo11n.pt` — **COCO-pretrained, not fine-tuned** |
| Depth | Depth Anything V2 Small, `src/smart_scene_analyzer/depth.py` |
| Confidence threshold | script default |
| Credits | 0 |

> ⚠️ **This frame undercuts the main purpose of the step, and that should be said plainly.**
> The step exists because every number so far came from the test split — same sensor
> family, same kind of room, same annotation process — and a webcam is none of those
> things. This frame **is** from the test split, so it is not the out-of-distribution
> check the step is for. It validates that detection and depth run on the same pixels and
> produce one fused answer; it does not tell us how the models behave on a different
> camera. Re-run with `--camera 0` on a real webcam to get the finding this step wants.

## What the frame produced

Seven detections, sorted near to far by relative inverse depth:

| # | Class (COCO) | Conf | Rel. depth | Reading |
|---|---|---|---|---|
| 1 | `dining table` | 0.38 | 4.461 | the round table — **duplicate of #2** |
| 2 | `dining table` | 0.51 | 4.013 | the same table again, slightly tighter box |
| 3 | `chair` | 0.89 | 3.469 | left-foreground chair, nearest chair to camera |
| 4 | `chair` | 0.49 | 3.014 | left chair, partly behind #3 |
| 5 | `chair` | 0.67 | 2.419 | chair behind the table |
| 6 | `chair` | 0.80 | 0.552 | chair far down the corridor, right |
| 7 | `chair` | 0.38 | 0.296 | furthest chair, right edge of the corridor |

**No detection returned a `none` depth value.** Every box enclosed pixels after clipping,
so none of the four degenerate cases in `depth.py` was exercised on this frame.

## Question 1 — are the boxes on the right things, and what is missing?

**Found, correctly:** the table and all four near chairs, plus the two corridor chairs.
Box placement is tight on the chairs and reasonable on the table.

**Wrong:** the table is detected **twice** (#1 and #2), two overlapping boxes for one
object at 0.38 and 0.51 confidence. Duplicate detections survive NMS when the two boxes
disagree enough on extent — here one includes the table's base and one does not.

**Missing, and this is the informative part.** Two objects in frame belong to our taxonomy
and were not detected, because COCO has no equivalent class:

- the **blue bin** beside the pillar → our `bin`
- the **doorway** at the right of the corridor → our `door`

Against our K=10 taxonomy, COCO-pretrained weights can find **five classes and cannot
find five**:

| Ours | COCO equivalent | Findable now? |
|---|---|---|
| `bed` | `bed` | yes |
| `chair` | `chair` | yes |
| `sofa` | `couch` | yes |
| `table` | `dining table` | partly — unreliable on coffee and side tables |
| `screen` | `tv`, `laptop` | yes |
| `bin` | — | **no** |
| `door` | — | **no** |
| `lamp` | — | **no** |
| `pillow` | — | **no** |
| `storage` | — | **no** |

**That gap is the argument for fine-tuning**, and it is visible in this one frame: a bin
and a door sit in plain sight and the model is structurally incapable of naming them.

> **Note for the course:** the prompt's own table lists eight classes (`bed`, `chair`,
> `sofa`, `table`, `tv`, `cabinet`, `door`, `lamp`) and says five of eight are findable.
> This project's taxonomy is **ten** classes (ADR 0002), with `storage` in place of
> `cabinet` and with `bin`, `pillow` and `screen` added. The ratio is five of ten here.

## Question 2 — is the near-to-far order right?

**Yes, on every pair, checked against the rendered depth map rather than assumed.**

The table reads nearest (4.46, 4.01) and it is nearest — it fills the lower-centre of the
frame. The three table-side chairs follow at 3.47, 3.01 and 2.42, in the order their
distance from the camera increases. The two corridor chairs sit far below at 0.552 and
0.296, and the corridor is unmistakably the furthest part of the scene.

The depth PNG agrees independently: the tabletop is the brightest region, the foreground
floor is bright, the pillar is mid-grey, and the corridor falls away to near-black.
**Brighter is nearer**, which is the convention `depth.py` claims.

The gap between the near group (2.4–4.5) and the corridor group (0.3–0.6) is large and in
the right direction. Ordering within the near group is the part that would be worth
distrusting if two values were close; here they are not.

## What a reader should not take from this

- **No percentage, and no rate.** Seven boxes on one frame.
- **No distance.** The depth column has no unit and no scale; it is comparable only within
  this frame.
- **No cross-frame comparison.** Running this on a second image produces values that
  cannot be compared with these.

## To re-run after training

```bash
uv run python scripts/check_live_capture.py --camera 0 \
  --weights runs/detect/runs/v1-baseline/weights/best.pt
```

Expect `bin`, `door`, `lamp`, `pillow` and `storage` to start appearing, and expect the
COCO classes to possibly get **worse** — 7,724 indoor images is a narrower world than
COCO, and narrowing is the trade. Both directions belong in
`docs/model-card-v1.md` under known failure modes.

Note the weights path is `runs/detect/runs/v1-baseline/...`, not `runs/v1-baseline/...`:
ultralytics prepends its own settings prefix to `--project`.
