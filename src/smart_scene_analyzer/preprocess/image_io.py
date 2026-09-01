"""Stage S0 — image bytes to `SourceImage`. Defines the `SOURCE` coordinate space.

**Single responsibility:** produce the one decoded, display-upright RGB array that every
later stage measures its coordinates against.

Three things happen here and nowhere else:

1. **A downscaling decode, not a decode-then-resize.** N7 requires a long edge of 3024 px
   or more to be handled without an out-of-memory failure; a 3024x4032 RGBA decode is
   roughly 48.8 MB resident before either model has loaded. The decoder is asked for a
   reduced image directly (`Image.draft` / `inSampleSize` / `CGImageSourceCreateThumbnail`
   depending on platform), targeting `DECODE_LONG_EDGE` (A1). Decoding full size and then
   resizing is the standard way to fail N7.
2. **EXIF orientation is applied.** `SOURCE` is display-upright, and no later module
   rotates anything. Skipping this makes boxes appear rotated 90 degrees on exactly the
   photographs taken in portrait, and on no others (A10).
3. **Alpha is dropped and channel order is fixed to RGB.** There is no RGBA and no BGR
   anywhere after this function.

The function takes **bytes or a file-like object**, not a path, so unit tests can feed it
an in-memory buffer and run with nothing on disk.

Planned public surface:

    def decode(
        data: bytes,
        long_edge: int,
    ) -> SourceImage
        \"\"\"Decode to uint8 [H_src, W_src, 3] RGB in SOURCE space, long edge <= `long_edge`,
        EXIF orientation applied. Records `original_size` (the file's grid) and
        `decode_scale` (SOURCE px per file px, <= 1.0) so the file's frame stays
        recoverable — see the F3 flag in docs/architecture.md §3.\"\"\"
"""
