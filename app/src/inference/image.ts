/**
 * URI -> decoded RGB pixels. The first stage of the whole pipeline: the picker and the
 * camera hand the rest of this app a **URI**, and neither on-device runtime decodes
 * images -- both `detector.ts` and `depth.ts` take typed-array tensors, not file paths.
 *
 * Two steps:
 *   1. Resize natively (expo-image-manipulator) so the long side is `longSide`, and
 *      render to a small JPEG. Native resize avoids decoding the full-resolution photo
 *      (routinely 3000px+ on a modern phone) into JS-reachable memory just to shrink it.
 *   2. Decode that small JPEG to raw RGB bytes (jpeg-js) so detector/depth preprocessing
 *      (letterbox + normalize) has a plain pixel buffer to work from.
 *
 * `Size` is defined here, minimally, because no shared geometry module exists yet --
 * `app/src/geometry/` (toScreen.ts and friends) is step 8's work. When that module
 * exists, this local `Size` should be replaced by its export rather than kept as a
 * second definition.
 *
 * The decoded image produced here is the **SOURCE** space for everything downstream:
 * the frame a detection box is eventually converted back into, and the frame the user
 * sees. It is NOT the original photo's resolution -- it is already resized to `longSide`.
 * Name any size/pixels coming out of this function as SOURCE, not "original".
 */
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import * as jpeg from 'jpeg-js';

/** Width/height in pixels of some named space. Which space is determined by context --
 * a shared, space-tagged version of this type arrives in step 8's geometry module. */
export interface Size {
  width: number;
  height: number;
}

/** Decoded RGB pixels, SOURCE space (see module header). Row-major, 3 bytes/pixel
 * (R, G, B — no alpha), length === size.width * size.height * 3. */
export interface RGBPixels {
  size: Size;
  data: Uint8Array;
}

/**
 * Resizes the image at `uri` so its long side is `longSide` pixels (short side scales
 * to preserve aspect ratio), then decodes it to RGB.
 *
 * `originalSize` is accepted to decide which of width/height is the "long side" to pin
 * -- expo-image-manipulator's `resize()` only computes the other dimension for you once
 * you tell it which one you're fixing; it doesn't accept "longest side = N" directly.
 */
export async function decodeImage(
  uri: string,
  originalSize: Size,
  longSide: number
): Promise<RGBPixels> {
  const isWide = originalSize.width >= originalSize.height;
  const targetSize = isWide
    ? { width: longSide, height: null }
    : { width: null, height: longSide };

  // Contextual API (ImageManipulator.manipulate), not the deprecated manipulateAsync --
  // confirmed against docs.expo.dev/versions/v58.0.0/sdk/imagemanipulator for this SDK.
  const context = ImageManipulator.manipulate(uri);
  context.resize(targetSize);
  const rendered = await context.renderAsync();
  const saved = await rendered.saveAsync({
    format: SaveFormat.JPEG,
    compress: 1, // no additional lossy compression beyond the resize itself
    base64: true,
  });

  if (!saved.base64) {
    throw new Error('decodeImage: expo-image-manipulator returned no base64 data');
  }

  const jpegBytes = base64ToBytes(saved.base64);

  // jpeg-js defaults to formatAsRGBA: true -- a 4-byte-per-pixel RGBA buffer -- even
  // though this is a 3-channel JPEG with no alpha channel of its own; it fabricates an
  // opaque alpha byte. useTArray: true asks for a Uint8Array instead of a Node Buffer
  // (React Native has no Buffer global). Drop the 4th byte per pixel below rather than
  // passing formatAsRGBA: false, so the exact shape jpeg-js hands back is visible here.
  const decoded = jpeg.decode(jpegBytes, { useTArray: true });
  const { width, height, data } = decoded;

  const rgb = new Uint8Array(width * height * 3);
  for (let src = 0, dst = 0; src < data.length; src += 4, dst += 3) {
    rgb[dst] = data[src];
    rgb[dst + 1] = data[src + 1];
    rgb[dst + 2] = data[src + 2];
  }

  return { size: { width, height }, data: rgb };
}

/**
 * Plain-JS base64 decoder. React Native / Hermes has neither a global `atob` nor a
 * `Buffer` global, and pulling in a decoding library for this one call would be a new
 * dependency for ~25 lines of arithmetic -- so it's written out instead of installed.
 */
function base64ToBytes(base64: string): Uint8Array {
  const alphabet =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const lookup = new Int8Array(128).fill(-1);
  for (let i = 0; i < alphabet.length; i++) {
    lookup[alphabet.charCodeAt(i)] = i;
  }

  const clean = base64.replace(/[\r\n]/g, '');
  const padding = clean.endsWith('==') ? 2 : clean.endsWith('=') ? 1 : 0;
  const cleanNoPad = padding === 0 ? clean : clean.slice(0, clean.length - padding);

  const outLength = Math.floor((cleanNoPad.length * 6) / 8);
  const out = new Uint8Array(outLength);

  let buffer = 0;
  let bitsInBuffer = 0;
  let outIndex = 0;
  for (let i = 0; i < cleanNoPad.length; i++) {
    const value = lookup[cleanNoPad.charCodeAt(i)];
    if (value === -1) continue; // skip any unexpected whitespace/padding
    buffer = (buffer << 6) | value;
    bitsInBuffer += 6;
    if (bitsInBuffer >= 8) {
      bitsInBuffer -= 8;
      out[outIndex++] = (buffer >> bitsInBuffer) & 0xff;
    }
  }

  return out;
}
