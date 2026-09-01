"""Operations on `xyxy` boxes, within a single coordinate space.

**Single responsibility:** box arithmetic that does not change coordinate space. Anything
that crosses a space belongs in `letterbox.py`, not here.

Every function takes and returns `xyxy` in absolute pixels, and **both the argument and
the result are in the same space** — whichever space the caller was already in. That is
the invariant that makes this module safe to use from both `SOURCE` and model-input code.

Planned public surface:

    def clip(box: BoxXYXY, size: Size) -> BoxXYXY
        \"\"\"Clamp to [0, w] x [0, h]. May return a zero-area box; the caller decides
        whether to drop it (contract C7 drops it).\"\"\"

    def inset(box: BoxXYXY, fraction: float, min_side_px: float = 1.0) -> BoxXYXY
        \"\"\"Shrink by `fraction` of each side, keeping the centre. Floor-clamped so each
        side keeps at least `min_side_px`. Used by the fusion reduction (A5): box corners
        are background in almost every box, and the centre is where the object is.\"\"\"

    def area(box: BoxXYXY) -> float
    def iou(a: BoxXYXY, b: BoxXYXY) -> float
    def intersect(a: BoxXYXY, b: BoxXYXY) -> BoxXYXY | None
        \"\"\"None when the boxes do not overlap — not a zero-area box, which would be
        indistinguishable from a legitimate degenerate one.\"\"\"

    def is_valid(box: BoxXYXY) -> bool
        \"\"\"x1 <= x2 and y1 <= y2 and no NaN. The invariant every Detection asserts.\"\"\"
"""
