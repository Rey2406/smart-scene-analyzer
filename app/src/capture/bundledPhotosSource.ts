/**
 * A capture source backed by the photos already bundled in app/assets/test-images/,
 * cycling through them one per call. Two jobs:
 *   1. The fallback Live mode offers when there is no camera AND no clip has been
 *      picked yet -- so the Live tab is never a dead end on a device with neither.
 *   2. The source the fake-timer unit test (live/liveLoop.test.ts) exercises the
 *      skip-vs-queue rule against, in spirit -- that test uses its own minimal stub
 *      rather than this class directly, so it has no dependency on expo-asset or the
 *      file system, but this is the same role: a StillCaptureSource that always has
 *      something to hand back, synchronously ready or not.
 *
 * Resolves each bundled module to a real file URI with expo-asset, same mechanism
 * detector.ts/depth.ts already use for the bundled model files. Caches each resolved
 * URI after its first resolution -- re-resolving a bundled asset on every capture would
 * cost a needless async round trip once a second.
 */
import { Asset } from 'expo-asset';

import type { StillCaptureSource } from './types';

const BUNDLED_MODULES: number[] = [
  require('../../assets/test-images/kv2_align_kv2_2014-12-18_11-47-06_260595134347_0000037.jpg'),
  require('../../assets/test-images/round4-landscape-4x3.jpg'),
  require('../../assets/test-images/round4-portrait-4x3.jpg'),
  require('../../assets/test-images/round4-portrait-16x9.jpg'),
];

export class BundledPhotosCaptureSource implements StillCaptureSource {
  private nextIndex = 0;
  private readonly resolvedUris = new Map<number, string>();

  async captureStill(): Promise<string> {
    const moduleId = BUNDLED_MODULES[this.nextIndex]!;
    this.nextIndex = (this.nextIndex + 1) % BUNDLED_MODULES.length;

    const cached = this.resolvedUris.get(moduleId);
    if (cached) return cached;

    const asset = Asset.fromModule(moduleId);
    await asset.downloadAsync();
    const uri = asset.localUri ?? asset.uri;
    if (!uri) {
      throw new Error('BundledPhotosCaptureSource: bundled image resolved to no local URI');
    }
    this.resolvedUris.set(moduleId, uri);
    return uri;
  }
}
