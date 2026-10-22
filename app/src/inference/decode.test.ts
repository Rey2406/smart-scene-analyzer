/**
 * Hand-built tensor, hand-computed expected boxes -- not a snapshot. See decode.ts's
 * header for the real on-device/Python evidence behind the channel-major, cxcywh,
 * normalized-[0,1] layout this test encodes.
 *
 * NOTE: no test runner is wired into app/ yet (no jest/jest-expo in package.json --
 * see the Round 1 report). This file is written in standard jest syntax, the Expo
 * default, so it runs unmodified once a runner is added; until then it was verified by
 * compiling this logic with `tsc` and executing it directly with `node` (see the Round
 * 1 report for the transcript).
 */
import { decodeDetections } from './decode';

describe('decodeDetections', () => {
  it('decodes cxcywh/normalized anchors above threshold, in channel-major layout, and drops low-confidence ones', () => {
    // 2 anchors, 7 features (4 box + 3 classes -- real model has 10 classes; this test
    // only exercises classes 0/1/2, i.e. labels 'bed'/'bin'/'chair').
    const anchors = 2;
    const features = 7;

    // Anchor 0: cx=0.5, cy=0.5, w=0.2, h=0.4, class scores [0.9, 0.1, 0.2] -> best
    // class 0 ("bed"), confidence 0.9, well above CONF_THRESHOLD (0.25).
    // Anchor 1: cx=0.3, cy=0.6, w=0.1, h=0.1, class scores [0.05, 0.05, 0.05] -> best
    // score 0.05, below CONF_THRESHOLD -- must be dropped.
    //
    // Flattened channel-major: raw[channel * anchors + anchor].
    const raw = new Float32Array([
      /* channel 0: cx     */ 0.5, 0.3,
      /* channel 1: cy     */ 0.5, 0.6,
      /* channel 2: w      */ 0.2, 0.1,
      /* channel 3: h      */ 0.4, 0.1,
      /* channel 4: bed    */ 0.9, 0.05,
      /* channel 5: bin    */ 0.1, 0.05,
      /* channel 6: chair  */ 0.2, 0.05,
    ]);

    const result = decodeDetections(raw, { anchors, features }, { width: 100, height: 100 });

    // Anchor 0, hand-computed in MODEL-INPUT pixels (100x100 canvas):
    //   x1 = (0.5 - 0.2/2) * 100 = 40
    //   y1 = (0.5 - 0.4/2) * 100 = 30
    //   x2 = (0.5 + 0.2/2) * 100 = 60
    //   y2 = (0.5 + 0.4/2) * 100 = 70
    // toBeCloseTo, not toEqual, on every float: `raw` is a Float32Array, so inputs like
    // 0.5/0.2/0.4 round to the nearest float32 on the way in and carry that rounding
    // through the arithmetic -- comparing for exact float64 equality would be comparing
    // against a precision the input never had.
    expect(result).toHaveLength(1);
    const [d] = result;
    expect(d.classId).toBe(0);
    expect(d.label).toBe('bed');
    expect(d.confidence).toBeCloseTo(0.9, 5);
    expect(d.box.x1).toBeCloseTo(40, 3);
    expect(d.box.y1).toBeCloseTo(30, 3);
    expect(d.box.x2).toBeCloseTo(60, 3);
    expect(d.box.y2).toBeCloseTo(70, 3);
  });

  it('picks the argmax class per anchor, not always class 0', () => {
    const anchors = 1;
    const features = 7;
    // cx=0.5, cy=0.5, w=0.2, h=0.2; class scores [0.1, 0.1, 0.95] -> best class 2 ("chair").
    const raw = new Float32Array([0.5, 0.5, 0.2, 0.2, 0.1, 0.1, 0.95]);

    const result = decodeDetections(raw, { anchors, features }, { width: 640, height: 640 });

    expect(result).toHaveLength(1);
    expect(result[0].classId).toBe(2);
    expect(result[0].label).toBe('chair');
    // x1 = (0.5 - 0.1) * 640 = 256, x2 = (0.5 + 0.1) * 640 = 384
    expect(result[0].box.x1).toBeCloseTo(256, 2);
    expect(result[0].box.y1).toBeCloseTo(256, 2);
    expect(result[0].box.x2).toBeCloseTo(384, 2);
    expect(result[0].box.y2).toBeCloseTo(384, 2);
  });

  it('throws if the layout implies a different class count than app/src/labels.ts', () => {
    const raw = new Float32Array([0, 0, 0, 0, 0]); // 1 anchor, 5 features (1 class)
    expect(() => decodeDetections(raw, { anchors: 1, features: 5 }, { width: 640, height: 640 })).toThrow();
  });
});
