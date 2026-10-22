/**
 * Hand-computed expected values -- not snapshots. See decode.test.ts's header for why:
 * no test runner (jest/jest-expo) is wired into app/ yet (Lesson 05 step 6's Round 1
 * report). Written in standard jest syntax for when one is added; verified in the
 * meantime by compiling with `tsc` and running directly with `node` (see the Round 2
 * report for the transcript).
 *
 * Every expected box below is arithmetic worked out by hand in the comment above it,
 * not a value copied from running the function once and trusting the output.
 */
import { modelInputPixelsToScreenPoints } from './toScreen';

describe('modelInputPixelsToScreenPoints', () => {
  it('identity: square image, square model input (no padding), square view (no view-level bars)', () => {
    // scale=1, padX=0, padY=0 -- model input IS the source image, pixel for pixel.
    // sourceImageSize === screenViewSize -- the view IS the image, point for point.
    // Undo-letterbox step changes nothing (padX/padY=0, scale=1); view-mapping step
    // changes nothing (displayScale=1, offsets=0). Box should pass through unchanged.
    const result = modelInputPixelsToScreenPoints(
      { x1: 10, y1: 20, x2: 50, y2: 60 },
      { scale: 1, padX: 0, padY: 0 },
      { width: 100, height: 100 },
      { width: 100, height: 100 }
    );
    expect(result).toEqual({ x1: 10, y1: 20, x2: 50, y2: 60 });
  });

  it('non-square padding: a box at the top edge of the CONTENT must not land at the top edge of the VIEW', () => {
    // SOURCE image: 300x100 (3:1 landscape). MODEL input: square 150x150.
    //   scale = min(150/300, 150/100) = min(0.5, 1.5) = 0.5
    //   scaledW = 300*0.5 = 150, scaledH = 100*0.5 = 50
    //   padX = floor((150-150)/2) = 0
    //   padY = floor((150-50)/2)  = 50   <- the vertical bars
    // So the image's CONTENT occupies MODEL-INPUT y in [50, 100]; y=50 is the content's
    // own top edge (source y=0).
    //
    // Box in MODEL-INPUT pixels, top edge exactly at the content's top (y1=50):
    //   { x1: 30, y1: 50, x2: 90, y2: 80 }
    //
    // Step 1 -- undo the MODEL letterbox (subtract pad, THEN divide by scale):
    //   sourceX1 = (30 - 0) / 0.5 = 60
    //   sourceY1 = (50 - 50) / 0.5 = 0     <- correctly recovers the image's true top
    //   sourceX2 = (90 - 0) / 0.5 = 180
    //   sourceY2 = (80 - 50) / 0.5 = 60
    //   (A bug that skips subtracting padY would compute sourceY1 = 50/0.5 = 100
    //   instead of 0 -- wrong, but invisible on the square identity test above, where
    //   padY is 0 either way. This is the case that catches it.)
    //
    // VIEW: square 150x150. imageAspect = 300/100 = 3; viewAspect = 150/150 = 1.
    // imageAspect > viewAspect -> fits to the view's WIDTH, bars above/below:
    //   displayScale = 150/300 = 0.5
    //   displayHeight = 100*0.5 = 50
    //   viewOffsetX = 0
    //   viewOffsetY = (150-50)/2 = 50
    //
    // Step 2 -- map SOURCE pixels into the view:
    //   screenX1 = 0  + 60*0.5  = 30
    //   screenY1 = 50 + 0*0.5   = 50   <- NOT 0: the view's own letterbox bar pushes
    //                                     even the image's true top edge away from the
    //                                     view's literal top edge. Proves both the
    //                                     model-letterbox subtraction AND the view's
    //                                     own fit composed correctly, instead of
    //                                     canceling out or being conflated.
    //   screenX2 = 0  + 180*0.5 = 90
    //   screenY2 = 50 + 60*0.5  = 80
    const result = modelInputPixelsToScreenPoints(
      { x1: 30, y1: 50, x2: 90, y2: 80 },
      { scale: 0.5, padX: 0, padY: 50 },
      { width: 300, height: 100 },
      { width: 150, height: 150 }
    );
    expect(result).toEqual({ x1: 30, y1: 50, x2: 90, y2: 80 });
    expect(result.y1).not.toBe(0); // the literal assertion the case is named for
  });

  it('a box at the exact image edge lands at the exact view edge', () => {
    // SOURCE image: 200x100 (2:1). MODEL input: square 100x100.
    //   scale = min(100/200, 100/100) = 0.5
    //   scaledW = 100, scaledH = 50
    //   padX = floor((100-100)/2) = 0
    //   padY = floor((100-50)/2)  = 25
    // Box = the full CONTENT bounding box in MODEL-INPUT pixels:
    //   { x1: padX, y1: padY, x2: padX+100, y2: padY+50 } = { x1:0, y1:25, x2:100, y2:75 }
    //
    // Step 1 -- undo letterbox:
    //   sourceX1 = (0-0)/0.5=0,   sourceY1=(25-25)/0.5=0
    //   sourceX2 = (100-0)/0.5=200, sourceY2=(75-25)/0.5=100
    //   -> (0, 0, 200, 100): the exact, full extent of the SOURCE image.
    //
    // VIEW: 200x100, SAME aspect as the source image (identity view mapping):
    //   imageAspect = 200/100 = 2 = viewAspect -> displayScale=1, both offsets=0.
    //
    // Step 2: screen box = source box, unchanged = (0, 0, 200, 100), i.e. exactly the
    // view's own four edges.
    const result = modelInputPixelsToScreenPoints(
      { x1: 0, y1: 25, x2: 100, y2: 75 },
      { scale: 0.5, padX: 0, padY: 25 },
      { width: 200, height: 100 },
      { width: 200, height: 100 }
    );
    expect(result).toEqual({ x1: 0, y1: 0, x2: 200, y2: 100 });
  });

  it('a box larger than the image clamps to the view bounds rather than overflowing', () => {
    // Same SOURCE/MODEL-input/letterbox as the previous case (200x100 source, square
    // 100x100 model input, scale=0.5, padX=0, padY=25), but a box that runs past the
    // content's own padded bounds in every direction:
    //   { x1: -10, y1: 20, x2: 120, y2: 90 }
    //
    // Step 1 -- undo letterbox (nothing clamped yet -- this step is pure arithmetic):
    //   sourceX1 = (-10-0)/0.5 = -20   (past the image's left edge)
    //   sourceY1 = (20-25)/0.5 = -10   (past the image's top edge)
    //   sourceX2 = (120-0)/0.5 = 240   (past the image's right edge, width=200)
    //   sourceY2 = (90-25)/0.5 = 130   (past the image's bottom edge, height=100)
    //
    // VIEW: identity, 200x100 (same as the previous case) -> displayScale=1, offsets=0.
    //   raw screen box, before clamping = (-20, -10, 240, 130).
    //
    // Clamp each coordinate to the view's own bounds, [0,200]x[0,100]:
    //   x1: clamp(-20, 0, 200) = 0      y1: clamp(-10, 0, 100) = 0
    //   x2: clamp(240, 0, 200) = 200    y2: clamp(130, 0, 100) = 100
    const result = modelInputPixelsToScreenPoints(
      { x1: -10, y1: 20, x2: 120, y2: 90 },
      { scale: 0.5, padX: 0, padY: 25 },
      { width: 200, height: 100 },
      { width: 200, height: 100 }
    );
    expect(result).toEqual({ x1: 0, y1: 0, x2: 200, y2: 100 });
  });
});
