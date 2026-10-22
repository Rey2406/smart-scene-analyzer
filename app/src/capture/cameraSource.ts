/**
 * A capture source backed by a live expo-camera `CameraView`. Thin by design: this
 * class owns nothing about permissions or device availability (the Live tab's own state
 * machine handles those three states -- see App.tsx) and nothing about the pipeline --
 * it hands back a still's file URI and nothing else.
 */
import type { CameraView } from 'expo-camera';

import type { StillCaptureSource } from './types';

export class CameraCaptureSource implements StillCaptureSource {
  constructor(private readonly getRef: () => CameraView | null) {}

  async captureStill(): Promise<string> {
    const camera = this.getRef();
    if (!camera) {
      throw new Error('CameraCaptureSource: camera view is not mounted');
    }
    // skipProcessing is deliberately NOT set -- this project's own letterbox
    // (preprocess.ts) assumes a correctly-oriented image, and skipProcessing trades
    // that correctness for speed this pipeline does not need (see expo-camera's own
    // CameraPictureOptions docs: skipProcessing causes "orientation uncertainty").
    const picture = await camera.takePictureAsync({ quality: 0.8 });
    if (!picture?.uri) {
      throw new Error('CameraCaptureSource: takePictureAsync returned no uri');
    }
    return picture.uri;
  }
}
