"""Aspect-preserving letterbox: `SOURCE` <-> a model-input space, both directions.

**Single responsibility:** compute the one transform that relates `SOURCE` to a model
input canvas, and apply it forward to boxes and inverse to boxes.

There are **two** transforms on this path, one per model, because the two models have
different input resolutions. Functions here take the transform as an argument and never
reach for a default; a function that guessed which model it was serving is the bug this
module is arranged to prevent.

Arithmetic (`docs/architecture.md` C3t):

    forward:  x_model  = x_source * scale + pad_x
              y_model  = y_source * scale + pad_y
    inverse:  x_source = (x_model - pad_x) / scale
              y_source = (y_model - pad_y) / scale

`scale = min(out_w / in_w, out_h / in_h)`; padding is split evenly between the two sides
of the short axis (assumption A4 — an export that pads one-sided changes this arithmetic).

Planned public surface:

    def make_transform(source: Size, out_w: int, out_h: int) -> LetterboxTransform
        \"\"\"Compute the SOURCE -> model-input transform for one model's canvas.\"\"\"

    def letterbox_image(
        pixels: npt.NDArray[np.uint8],       # [H_src, W_src, 3] RGB, SOURCE space
        transform: LetterboxTransform,
        pad_value: tuple[int, int, int],
    ) -> npt.NDArray[np.uint8]               # [out_h, out_w, 3] RGB, model-input space

    def letterbox_forward(box: BoxXYXY, transform: LetterboxTransform) -> BoxXYXY
        \"\"\"xyxy absolute px, SOURCE -> xyxy absolute px, model-input. Unclipped.\"\"\"

    def letterbox_inverse(box: BoxXYXY, transform: LetterboxTransform) -> BoxXYXY
        \"\"\"xyxy absolute px, model-input -> xyxy absolute px, SOURCE. Unclipped.\"\"\"

    def content_rect(transform: LetterboxTransform) -> BoxXYXY
        \"\"\"The un-padded region of the model-input canvas, xyxy in model-input px.
        Everything outside it is padding and must not be sampled (see DepthMap.valid).\"\"\"

Both box functions return **unclipped** boxes. Clipping to the image is a separate,
deliberate step that happens once, at contract boundary C7, in `inference.decode`.
"""
