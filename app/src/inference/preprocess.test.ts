/**
 * Hand-computed expected tensors, not snapshots -- see preprocess.ts's header for the
 * NCHW layout evidence and the letterbox/padding rationale this test encodes.
 *
 * NOTE: no test runner is wired into app/ yet (no jest/jest-expo in package.json --
 * same gap decode.test.ts already recorded). This file is standard jest syntax, the
 * Expo default, so it runs unmodified once a runner is added; until then it was
 * verified by compiling with `tsc` and executing the compiled JS directly with `node`.
 */
import { toModelInput } from './preprocess';
import type { RGBPixels } from './image';

describe('toModelInput', () => {
  it('normalizes correctly with no letterboxing needed (source size === target size)', () => {
    // 2x2 source, R=G=B per pixel so one set of hand-computed values covers all three
    // channels. Row-major, 3 bytes/pixel, matching image.ts's RGBPixels contract:
    //   (x0,y0)=0  (x1,y0)=128
    //   (x0,y1)=255 (x1,y1)=64
    const pixels: RGBPixels = {
      size: { width: 2, height: 2 },
      data: new Uint8Array([
        0, 0, 0, /*     */ 128, 128, 128,
        255, 255, 255, /**/ 64, 64, 64,
      ]),
    };

    const norm = { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] } as const;
    const result = toModelInput(pixels, { width: 2, height: 2 }, norm);

    // Target == source size -> scale 1, no padding.
    expect(result.scale).toBeCloseTo(1, 10);
    expect(result.padX).toBe(0);
    expect(result.padY).toBe(0);

    // (v/255 - 0.5) / 0.5 == 2v/255 - 1, hand-computed per source pixel:
    //   v=0   -> -1
    //   v=128 -> 2*128/255 - 1 = 0.00392156862745097
    //   v=255 -> 1
    //   v=64  -> 2*64/255 - 1 = -0.4980392156862745
    // NCHW, so R/G/B planes are identical here (every pixel has R=G=B). Each plane is
    // laid out row-major in (ty,tx) order: (0,0)=0, (0,1)=128, (1,0)=255, (1,1)=64.
    const expectedPlane = [-1, 0.00392156862745097, 1, -0.4980392156862745];

    expect(result.data).toHaveLength(3 * 2 * 2);
    for (let c = 0; c < 3; c++) {
      const start = c * 4;
      for (let i = 0; i < 4; i++) {
        expect(result.data[start + i]).toBeCloseTo(expectedPlane[i]!, 6);
      }
    }
  });

  it('letterboxes a narrow source into a wider target, padding the slack axis, no stretch', () => {
    // 1x2 source (width=1, height=2): row0 = (10,20,30), row1 = (40,50,60).
    const pixels: RGBPixels = {
      size: { width: 1, height: 2 },
      data: new Uint8Array([10, 20, 30, 40, 50, 60]),
    };

    // Identity normalization isolates the letterbox geometry from the normalization
    // math (already covered by the previous test).
    const norm = { mean: [0, 0, 0], std: [1, 1, 1] } as const;
    const result = toModelInput(pixels, { width: 2, height: 2 }, norm);

    // scale = min(2/1, 2/2) = 1 -- limited by height, not width.
    // scaledW = 1, scaledH = 2 -> padX = floor((2-1)/2) = 0, padY = floor((2-2)/2) = 0.
    // So column 0 is the real image, column 1 is pure letterbox bar.
    expect(result.scale).toBeCloseTo(1, 10);
    expect(result.padX).toBe(0);
    expect(result.padY).toBe(0);

    const planeSize = 4; // 2x2
    // R plane, (ty,tx) row-major: (0,0)=10/255 real, (0,1)=0 pad,
    //                             (1,0)=40/255 real, (1,1)=0 pad.
    const rPlane = [10 / 255, 0, 40 / 255, 0];
    const gPlane = [20 / 255, 0, 50 / 255, 0];
    const bPlane = [30 / 255, 0, 60 / 255, 0];
    const expected = [...rPlane, ...gPlane, ...bPlane];

    expect(result.data).toHaveLength(3 * planeSize);
    for (let i = 0; i < expected.length; i++) {
      expect(result.data[i]).toBeCloseTo(expected[i]!, 6);
    }
  });
});
