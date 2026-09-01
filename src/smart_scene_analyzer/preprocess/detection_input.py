"""Stage S1 — `SourceImage` to `DetectionInput` (contract C3).

**Single responsibility:** build the detection model's input tensor and the
`SOURCE` -> `DET_INPUT` transform that goes with it, as one inseparable pair.

The tensor and the transform are returned together because they are only correct together.
A tensor without its transform cannot have its boxes brought home; a transform without its
tensor is a scale factor for an image nobody built.

Layout: `[1, DET_SIZE, DET_SIZE, 3]`, **NHWC**. This is not the depth model's layout, and
the difference is deliberate — the two runtimes differ. dtype, the quantization scale and
zero point, and whether values are scaled to `[0, 1]` before quantization are fixed by the
int8 export and read from `model_contract.json`, never written from memory.

Padding uses `pad_value` (A4, provisionally `(114, 114, 114)`), split evenly on both sides
of the short axis.

Planned public surface:

    def build(
        image: SourceImage,
        size: int,
        pad_value: tuple[int, int, int],
    ) -> DetectionInput
        \"\"\"Letterbox and quantize to [1, size, size, 3] NHWC in DET_INPUT space, with the
        SOURCE -> DET_INPUT LetterboxTransform attached.\"\"\"
"""
