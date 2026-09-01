"""Load and validate C12 fixtures from JSON.

**Single responsibility:** parse the shared JSON into the project's types, and reject a
fixture that does not state everything the port also needs.

Validation is strict on purpose. A fixture missing its `transform`, its `source_size`, or
its declared coordinate space would still load in one of the two languages and be
interpreted by default in the other — which is a silent divergence, and silent divergence
is the specific thing this whole arrangement exists to prevent. So the loader fails on a
missing field rather than defaulting it.

Fixture shape (contract C12):

    {
      "depth_map":   {"size": [h, w], "values": [...]},   # row-major, DEPTH_INPUT space,
                                                          # relative inverse depth,
                                                          # larger is nearer, no unit
      "transform":   {"scale", "pad_x", "pad_y", "in_w", "in_h", "out_w", "out_h"},
      "valid":       [x1, y1, x2, y2],                    # DEPTH_INPUT px, un-padded rect
      "detections":  [{"bbox": [x1,y1,x2,y2], "class_id", "confidence"}],  # SOURCE px
      "source_size": {"w", "h"},
      "expected":    {"detections": [{"relative_depth", "relative_depth_normalized",
                                      "depth_rank", "depth_spread",
                                      "depth_sample_count"}]},
      "tolerance":   {"relative_depth": 1e-5, "normalized": 1e-5}
    }

`null` is the encoding for "no depth derivable" and it must round-trip as `null`. A fixture
that encodes it as `0` asserts that the object is at the far end of the scale, which is a
different claim and a passing test for the wrong behaviour.

Both languages parse numbers to 64-bit floats and compare within the fixture's stated
`tolerance`. The tolerance is part of the fixture rather than the test, so that widening it
is a visible change to a committed file.

Planned public surface:

    def load(path: Path) -> FusionFixture
    def load_all(directory: Path) -> list[FusionFixture]
    def validate(raw: Mapping[str, object]) -> list[str]
        \"\"\"-> human-readable problems. Empty list means the fixture is loadable by both
        the Python suite and the TypeScript one.\"\"\"
"""
