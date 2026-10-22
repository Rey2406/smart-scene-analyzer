/**
 * Loads the SAME plain-JSON fixture file `tests/test_fusion.py`'s
 * `test_shared_fusion_fixture_matches_python` loads (`tests/fixtures/fusion_cases.json`,
 * contract C12, N21) and asserts this port's output against it. This is the one file
 * that keeps the Python reference and this TypeScript port from silently diverging --
 * a disagreement here is the finding, never a reason to adjust the fixture.
 *
 * NOTE: no test runner (jest/jest-expo) is wired into app/ yet -- see
 * decode.test.ts / toScreen.test.ts's own headers for the same situation. Written in
 * standard jest syntax so it runs unmodified once a runner is added; verified in the
 * meantime by compiling with `tsc` and running the compiled JS directly with `node`
 * (see this round's report for the transcript).
 */
import * as fs from 'fs';
import * as path from 'path';

import { fuse, contentRect, MIXED_REGION_SPREAD_THRESHOLD, type DepthMap, type LetterboxTransform } from './index';

const FUSION_CASES_PATH = path.join(__dirname, '../../../tests/fixtures/fusion_cases.json');

interface RawTransform {
  scale: number;
  pad_x: number;
  pad_y: number;
  in_w: number;
  in_h: number;
  out_w: number;
  out_h: number;
}

interface RawDetection {
  bbox: [number, number, number, number];
  class_id: number;
  confidence: number;
}

interface RawExpectedDetection {
  relative_depth: number | null;
  relative_depth_normalized: number | null;
  depth_rank: number | null;
  depth_sample_count: number;
  depth_quality?: 'ok' | 'unavailable' | 'fallback_full_box' | 'mixed_region';
}

interface RawCase {
  name: string;
  depth_map: { size: [number, number]; values: (number | string)[][] };
  transform: RawTransform;
  detections: RawDetection[];
  expected: { detections: RawExpectedDetection[] };
  tolerance: { relative_depth: number; normalized: number };
}

/** Plain JSON has no literal NaN/Infinity -- JS `JSON.parse` rejects the bare tokens
 * outright, so non-finite depth samples are encoded as the strings "NaN" / "Infinity" /
 * "-Infinity" instead (see tests/test_fusion.py's `_parse_depth_values`, the same
 * convention on the Python side of this SAME fixture file). */
function parseDepthValue(v: number | string): number {
  if (typeof v === 'number') return v;
  if (v === 'NaN') return NaN;
  if (v === 'Infinity') return Infinity;
  if (v === '-Infinity') return -Infinity;
  throw new Error(`unrecognized depth value sentinel: ${v}`);
}

function loadCases(): RawCase[] {
  return JSON.parse(fs.readFileSync(FUSION_CASES_PATH, 'utf-8'));
}

function toDepthMap(raw: RawCase): DepthMap {
  const [w, h] = raw.depth_map.size;
  const flat = new Float32Array(w * h);
  raw.depth_map.values.forEach((row, y) => {
    row.forEach((v, x) => {
      flat[y * w + x] = parseDepthValue(v);
    });
  });
  const t = raw.transform;
  const transform: LetterboxTransform = {
    scale: t.scale,
    padX: t.pad_x,
    padY: t.pad_y,
    inW: t.in_w,
    inH: t.in_h,
    outW: t.out_w,
    outH: t.out_h,
  };
  // `valid` is NOT stored in the fixture (contract C12) -- derived from `transform`
  // alone, exactly like the Python loader's `content_rect(transform)` call.
  return { values: flat, width: w, height: h, transform, valid: contentRect(transform) };
}

describe('fuse (shared fixture: tests/fixtures/fusion_cases.json)', () => {
  const cases = loadCases();

  it('loaded at least the 6 minimum-coverage cases', () => {
    expect(cases.length).toBeGreaterThanOrEqual(6);
  });

  it(`MIXED_REGION_SPREAD_THRESHOLD matches the Python reference (0.3)`, () => {
    expect(MIXED_REGION_SPREAD_THRESHOLD).toBe(0.3);
  });

  for (const raw of cases) {
    it(raw.name, () => {
      const depthMap = toDepthMap(raw);
      const detections = raw.detections.map((d) => ({
        box: { x1: d.bbox[0], y1: d.bbox[1], x2: d.bbox[2], y2: d.bbox[3] },
        classId: d.class_id,
      }));

      const fused = fuse(detections, depthMap);
      expect(fused).toHaveLength(raw.expected.detections.length);

      fused.forEach((actual, i) => {
        const expected = raw.expected.detections[i]!;
        const tol = raw.tolerance.relative_depth;
        const normTol = raw.tolerance.normalized;

        // No NaN ever reaches a caller -- either a well-defined number, or null.
        if (actual.relativeDepth !== null) expect(Number.isFinite(actual.relativeDepth)).toBe(true);
        if (actual.relativeDepthNormalized !== null) {
          expect(Number.isFinite(actual.relativeDepthNormalized)).toBe(true);
        }

        if (expected.relative_depth === null) {
          expect(actual.relativeDepth).toBeNull();
        } else {
          expect(actual.relativeDepth).not.toBeNull();
          expect(Math.abs(actual.relativeDepth! - expected.relative_depth)).toBeLessThanOrEqual(tol);
        }

        if (expected.relative_depth_normalized === null) {
          expect(actual.relativeDepthNormalized).toBeNull();
        } else {
          expect(actual.relativeDepthNormalized).not.toBeNull();
          expect(
            Math.abs(actual.relativeDepthNormalized! - expected.relative_depth_normalized)
          ).toBeLessThanOrEqual(normTol);
        }

        expect(actual.depthRank).toBe(expected.depth_rank);
        expect(actual.depthSampleCount).toBe(expected.depth_sample_count);
        if (expected.depth_quality !== undefined) {
          expect(actual.depthQuality).toBe(expected.depth_quality);
        }
      });
    });
  }
});
