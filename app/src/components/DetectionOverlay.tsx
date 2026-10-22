/**
 * The overlay layer: detection boxes + class labels + a depth legend, drawn on top of
 * the displayed image. Composition contract -- this component draws ONLY the overlay
 * (absolutely-positioned boxes, legend, and the loading/error/empty placeholders); the
 * caller stacks it over an `<Image>` of the SAME `screenViewSize`, e.g.:
 *
 *   <View style={{ width, height }}>
 *     <Image source={...} resizeMode="contain" style={StyleSheet.absoluteFill} />
 *     <DetectionOverlay state={...} letterbox={...} sourceImageSize={...}
 *                        screenViewSize={{ width, height }} />
 *   </View>
 *
 * `resizeMode="contain"` on that `<Image>` is not a styling preference -- it is the
 * fit-within behavior `toScreen.ts` assumes (see that file's module header). A
 * `cover`/crop mode here would make the image and the overlay disagree about where the
 * picture actually is.
 *
 * COORDINATE ARITHMETIC LIVES IN toScreen.ts, NOT HERE. This file calls
 * `modelInputPixelsToScreenPoints` once per box and renders the result; it does not add,
 * subtract, multiply, or divide a coordinate anywhere in its own render logic.
 *
 * DEPTH: conveyed by ORDERING (nearer boxes stack on top, nearer labels sort first in
 * the legend's reading order) and SHADING (opacity + border weight scale with how near
 * a detection ranks among the others in THIS frame) -- never by a printed number.
 * `relativeDepth` is the model's raw relative-inverse-depth value (larger = nearer,
 * unitless, comparable only within one image -- see depth.ts's header); it is read here
 * ONLY to rank detections against each other, and the number itself is never rendered,
 * logged to the UI, or interpolated into any string. Absolutely no distance, unit, or
 * physical-measurement-shaped value belongs in this file -- units_guard is a backstop
 * for identifiers and prose in `app/`, not a substitute for this rule; see
 * app/CLAUDE.md's artifact-contract section.
 */
import { StyleSheet, Text, View, ActivityIndicator } from 'react-native';

import type { Label } from '../labels';
import {
  modelInputPixelsToScreenPoints,
  type BoxXYXYModelInputPixels,
} from '../geometry/toScreen';

/** One detection, ready to draw: a MODEL-INPUT box, its class label, and the model's
 * raw relative-inverse-depth value for ranking against its frame-mates. Not a distance. */
export interface OverlayDetection {
  box: BoxXYXYModelInputPixels;
  label: Label;
  confidence: number;
  /** Relative inverse depth: larger = nearer, unitless, comparable only within this
   * same detection set. Used here only to RANK -- never displayed as a number. */
  relativeDepth: number;
}

/** Mirrors the pipeline's real states for the detector. "Zero detections" is a fourth,
 * SUCCESSFUL outcome of 'ready' (an empty array), not a state of its own -- it must
 * still read differently on screen from both 'loading' and 'error' (see the empty-list
 * branch in the component below). */
export type DetectionOverlayState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; detections: OverlayDetection[] };

export interface DetectionOverlayProps {
  state: DetectionOverlayState;
  /** The exact letterbox `toModelInput` (preprocess.ts) returned for this image --
   * never recomputed here. */
  letterbox: { scale: number; padX: number; padY: number };
  sourceImageSize: { width: number; height: number };
  /** The displaying view's real size, dp, from its own `onLayout` -- never assumed. */
  screenViewSize: { width: number; height: number };
}

/** Shared accent color for every box -- class is conveyed by the text label, not by a
 * second color channel, so color stays free to mean exactly one thing: depth rank. */
const BOX_COLOR = '38, 166, 154'; // teal, as an "r, g, b" string for template use in rgba()

/** Floor and ceiling for the opacity/border-weight ramp -- the farthest object in a
 * frame is still visible (never fully transparent; this is a ranking, not a fade-out),
 * and the nearest is never heavier than this. Tuned by eye, not measured -- a styling
 * choice, not a fact. */
const OPACITY_RANGE: [number, number] = [0.38, 1.0]; // [farthest, nearest]
const BORDER_WIDTH_RANGE: [number, number] = [1, 3.5]; // [farthest, nearest]

interface RankedDetection extends OverlayDetection {
  /** 0 = nearest in this frame. Used only to position in the ORDERING (stack order,
   * legend reading order) and to compute SHADING below -- never displayed. */
  rank: number;
  opacity: number;
  borderWidth: number;
}

/** Ranks detections nearest-first by `relativeDepth` and derives a shading value for
 * each from its rank among the others -- RELATIVE to this frame's own min/max, so a
 * room with one close object and a room with ten far ones both still show visible
 * contrast. A single detection ranks full-strength (nothing to be relatively farther
 * than). Pure: no rendering, no side effects -- the numbers it touches never leave this
 * function as text. */
function rankByDepth(detections: OverlayDetection[]): RankedDetection[] {
  const sorted = [...detections].sort((a, b) => b.relativeDepth - a.relativeDepth); // nearer (larger) first

  if (sorted.length <= 1) {
    const [, nearestOpacity] = OPACITY_RANGE;
    const [, nearestBorder] = BORDER_WIDTH_RANGE;
    return sorted.map((d, rank) => ({ ...d, rank, opacity: nearestOpacity, borderWidth: nearestBorder }));
  }

  const nearest = sorted[0]!.relativeDepth;
  const farthest = sorted[sorted.length - 1]!.relativeDepth;
  const spread = nearest - farthest;

  return sorted.map((d, rank) => {
    // 1 = nearest, 0 = farthest, within THIS frame's own range.
    const nearness = spread > 0 ? (d.relativeDepth - farthest) / spread : 1;
    const opacity = lerp(OPACITY_RANGE[0], OPACITY_RANGE[1], nearness);
    const borderWidth = lerp(BORDER_WIDTH_RANGE[0], BORDER_WIDTH_RANGE[1], nearness);
    return { ...d, rank, opacity, borderWidth };
  });
}

function lerp(min: number, max: number, t: number): number {
  return min + (max - min) * t;
}

export function DetectionOverlay({ state, letterbox, sourceImageSize, screenViewSize }: DetectionOverlayProps) {
  if (state.status === 'loading') {
    return (
      <View style={[styles.placeholder, styles.placeholderLoading]} pointerEvents="none">
        <ActivityIndicator color="#5b6b75" />
        <Text style={styles.placeholderText}>Scanning the scene{'…'}</Text>
      </View>
    );
  }

  if (state.status === 'error') {
    return (
      <View style={[styles.placeholder, styles.placeholderError]} pointerEvents="none">
        <Text style={styles.placeholderIcon}>{'⚠️'}</Text>
        <Text style={[styles.placeholderText, styles.placeholderTextError]}>Detection model failed to load</Text>
        <Text style={styles.placeholderSubtext}>{state.message}</Text>
      </View>
    );
  }

  const { detections } = state;

  if (detections.length === 0) {
    return (
      <View style={[styles.placeholder, styles.placeholderEmpty]} pointerEvents="none">
        <Text style={styles.placeholderIcon}>{'✓'}</Text>
        <Text style={styles.placeholderText}>No objects detected</Text>
        <Text style={styles.placeholderSubtext}>The scan finished cleanly -- nothing met the confidence threshold.</Text>
      </View>
    );
  }

  const ranked = rankByDepth(detections);

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Render FARTHEST first, NEAREST last -- later siblings stack on top in React
          Native's default paint order, so the ordering below IS the depth ordering:
          a near box visually occludes a far one the way the real scene would. */}
      {[...ranked].reverse().map((d, i) => {
        const screen = modelInputPixelsToScreenPoints(d.box, letterbox, sourceImageSize, screenViewSize);
        const left = screen.x1;
        const top = screen.y1;
        const width = Math.max(0, screen.x2 - screen.x1);
        const height = Math.max(0, screen.y2 - screen.y1);

        return (
          <View
            key={`${d.label}-${d.rank}-${i}`}
            style={[
              styles.box,
              {
                left,
                top,
                width,
                height,
                borderWidth: d.borderWidth,
                borderColor: `rgba(${BOX_COLOR}, ${d.opacity})`,
                zIndex: d.rank, // explicit, in case the platform ever reorders by zIndex
              },
            ]}
          >
            <View style={[styles.labelChip, { backgroundColor: `rgba(${BOX_COLOR}, ${d.opacity})` }]}>
              <Text style={styles.labelText}>{d.label}</Text>
            </View>
          </View>
        );
      })}

      <Legend />
    </View>
  );
}

/** Says "nearer" / "farther" in words, with two swatches at the same opacity/border
 * extremes the boxes themselves use -- so the legend and the boxes read as one scale,
 * without either one ever stating a number. */
function Legend() {
  return (
    <View style={styles.legend} pointerEvents="none">
      <View style={styles.legendItem}>
        <View style={[styles.legendSwatch, { opacity: OPACITY_RANGE[1], borderWidth: BORDER_WIDTH_RANGE[1] }]} />
        <Text style={styles.legendText}>nearer</Text>
      </View>
      <View style={styles.legendItem}>
        <View style={[styles.legendSwatch, { opacity: OPACITY_RANGE[0], borderWidth: BORDER_WIDTH_RANGE[0] }]} />
        <Text style={styles.legendText}>farther</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderRadius: 4,
    justifyContent: 'flex-start',
    alignItems: 'flex-start',
  },
  labelChip: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderTopLeftRadius: 3,
    borderBottomRightRadius: 6,
  },
  labelText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '700',
  },
  legend: {
    position: 'absolute',
    right: 8,
    bottom: 8,
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.85)',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    gap: 14,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendSwatch: {
    width: 14,
    height: 14,
    borderRadius: 3,
    borderColor: `rgb(${BOX_COLOR})`,
    backgroundColor: `rgb(${BOX_COLOR})`,
  },
  legendText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#2b2b2b',
  },
  placeholder: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  placeholderLoading: {
    backgroundColor: 'rgba(245,247,248,0.9)',
  },
  placeholderError: {
    backgroundColor: 'rgba(255,235,235,0.95)',
  },
  placeholderEmpty: {
    backgroundColor: 'rgba(235,250,240,0.9)',
  },
  placeholderIcon: {
    fontSize: 28,
    marginBottom: 8,
  },
  placeholderText: {
    fontSize: 15,
    fontWeight: '700',
    color: '#2b2b2b',
    textAlign: 'center',
  },
  placeholderTextError: {
    color: '#8a2b2b',
  },
  placeholderSubtext: {
    marginTop: 4,
    fontSize: 12,
    color: '#5b6b75',
    textAlign: 'center',
  },
});
