/**
 * A capture source backed by a recorded clip, for the iOS Simulator (no camera -- see
 * App.tsx's three-state permission UI). Produces a still exactly the way the camera
 * source does: a file URI, nothing decoded here.
 *
 * Per round 4 of resources/prompts/06-live-mode.md:
 *   1. `player.generateThumbnailsAsync(t)` -- a native thumbnail at time `t` (seconds).
 *   2. `ImageManipulator.manipulate(thumbnail).renderAsync()` -- expo-image-manipulator's
 *      contextual API accepts a thumbnail SharedRef directly (it extends
 *      `SharedRef<'image'>`, the same type `manipulate()`'s `source` parameter declares --
 *      confirmed against node_modules/expo-image-manipulator/src/ImageManipulator.types.ts),
 *      so no intermediate file write is needed just to hand the thumbnail off.
 *   3. `.saveAsync()` -- writes the actual file and returns its URI.
 * `t` advances by one interval per capture and wraps to 0 at the clip's own duration, so
 * a short clip loops rather than running out of frames.
 *
 * Does NOT decode pixels, and does NOT call analyzeImage.ts itself -- the Live tab hands
 * this source's returned URI to the SAME pipeline function Photo mode calls.
 */
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import type { VideoPlayer } from 'expo-video';

import type { StillCaptureSource } from './types';

export class ClipCaptureSource implements StillCaptureSource {
  private t = 0;

  constructor(
    private readonly player: VideoPlayer,
    /** The Live loop's own interval, in seconds (the loop runs in ms; this class only
     * needs the clip-time step, so the conversion happens once at construction). */
    private readonly stepSeconds: number
  ) {}

  async captureStill(): Promise<string> {
    const { duration } = this.player;
    if (!duration || duration <= 0) {
      throw new Error(
        'ClipCaptureSource: player.duration is not known yet -- wait for status "readyToPlay" before starting capture'
      );
    }

    const [thumbnail] = await this.player.generateThumbnailsAsync(this.t);
    if (!thumbnail) {
      throw new Error(`ClipCaptureSource: generateThumbnailsAsync(${this.t}) returned no thumbnail`);
    }

    const rendered = await ImageManipulator.manipulate(thumbnail).renderAsync();
    const saved = await rendered.saveAsync({ format: SaveFormat.JPEG, compress: 1 });

    // Advance AFTER a successful capture, wrapping at the clip's own duration -- a
    // short clip loops instead of the next call failing past the end of the media.
    this.t = (this.t + this.stepSeconds) % duration;

    return saved.uri;
  }
}
