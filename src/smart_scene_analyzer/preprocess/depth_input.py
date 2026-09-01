"""Stage S2 — `SourceImage` to `DepthInput` (contract C4).

**Single responsibility:** build the depth model's input tensor and the
`SOURCE` -> `DEPTH_INPUT` transform that goes with it.

Layout: `[1, 3, DEPTH_SIZE, DEPTH_SIZE]` float32, **NCHW** — channel-first, and therefore
**not** the detection model's NHWC layout. `[1, 3, 518, 518]` float32 NCHW and
`[1, 518, 518, 3]` uint8 NHWC are not interchangeable, and a silent swap produces a depth
map that looks like a depth map and is wrong everywhere.

The transform this returns is **not** `DetectionInput.transform`. Different input size
means a different scale and different pads. The fusion layer takes this one, and taking
the other is the single most expensive mistake available on this path — it returns an
in-range number for every object.

Channel normalization constants are export-owned (C11), provisionally ImageNet (A3).

Planned public surface:

    def build(
        image: SourceImage,
        size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        pad_value: tuple[int, int, int],
    ) -> DepthInput
        \"\"\"Letterbox and normalize to [1, 3, size, size] float32 NCHW in DEPTH_INPUT
        space, with the SOURCE -> DEPTH_INPUT LetterboxTransform attached.\"\"\"
"""
