/**
 * Hand-computed IoU and expected keep/drop outcomes -- not a snapshot. No test runner
 * is wired into app/ yet (see decode.test.ts's header and the Round 1 report); written
 * in standard jest syntax for when one is added, verified in the meantime by compiling
 * with `tsc` and running directly with `node`.
 */
import { nms, NMS_IOU_THRESHOLD } from './nms';
import type { DetectionModelInputPixels } from './decode';

function det(
  box: { x1: number; y1: number; x2: number; y2: number },
  classId: number,
  confidence: number
): DetectionModelInputPixels {
  return { box, classId, label: 'bed', confidence };
}

describe('nms', () => {
  it('collapses two heavily overlapping SAME-class boxes to one, keeping the higher-confidence box', () => {
    // A: [0,0]-[10,10] (area 100), conf 0.9.
    // B: [1,1]-[11,11] (area 100), conf 0.8.
    // Intersection: [1,1]-[10,10] -> 9x9 = 81. Union = 100 + 100 - 81 = 119.
    // IoU = 81/119 ≈ 0.6807, which is > 0.45 (NMS_IOU_THRESHOLD) -> B is suppressed.
    const a = det({ x1: 0, y1: 0, x2: 10, y2: 10 }, 0, 0.9);
    const b = det({ x1: 1, y1: 1, x2: 11, y2: 11 }, 0, 0.8);

    const result = nms([b, a], NMS_IOU_THRESHOLD); // order shouldn't matter -- nms sorts internally

    expect(result).toHaveLength(1);
    expect(result[0]).toBe(a); // the higher-confidence box survives
  });

  it('keeps two heavily overlapping boxes of DIFFERENT classes -- NMS is per class, not global', () => {
    const a = det({ x1: 0, y1: 0, x2: 10, y2: 10 }, 0, 0.9); // e.g. "bed"
    const b = det({ x1: 1, y1: 1, x2: 11, y2: 11 }, 1, 0.8); // e.g. "bin" -- same geometry, different class

    const result = nms([a, b], NMS_IOU_THRESHOLD);

    expect(result).toHaveLength(2);
    expect(result).toContain(a);
    expect(result).toContain(b);
  });

  it('returns an empty list for an empty input -- a successful result, not an error', () => {
    expect(nms([], NMS_IOU_THRESHOLD)).toEqual([]);
  });

  it('keeps both same-class boxes when they barely overlap (IoU below threshold)', () => {
    // A: [0,0]-[10,10] (area 100). B: [9,0]-[19,10] (area 100).
    // Intersection: [9,0]-[10,10] -> 1x10 = 10. Union = 100+100-10 = 190. IoU ≈ 0.0526 < 0.45.
    const a = det({ x1: 0, y1: 0, x2: 10, y2: 10 }, 0, 0.9);
    const b = det({ x1: 9, y1: 0, x2: 19, y2: 10 }, 0, 0.8);

    const result = nms([a, b], NMS_IOU_THRESHOLD);

    expect(result).toHaveLength(2);
  });
});
