"""The shared fixture set — the mechanism that keeps two implementations honest.

**Single responsibility:** define and load the fixtures that both the Python reference and
the TypeScript port are tested against (contract C12, requirement N21).

This package ships inside `src/` rather than living only under `tests/` because it is not a
test detail. It is the only mechanism by which one algorithm implemented twice, in two
languages, stays the same algorithm. `src/smart_scene_analyzer/fusion.py` and
`app/src/fusion/index.ts` agree because they are checked against the same bytes.

Three rules, and each one has been violated in some project before:

1. **Plain JSON. No pickles, no `.npy`.** A fixture the TypeScript suite cannot load tests
   half the system, and there is no warning when that happens — the Python suite still
   goes green.
2. **The fixture states its coordinate spaces.** Detections in `SOURCE`, the depth map and
   its transform in `DEPTH_INPUT`. A fixture that omits which space a box is in is a
   fixture that will be read wrong by exactly one of the two implementations.
3. **When the two disagree, the fixture does not move.** The disagreement is the finding:
   fix the port, or record why the reference is wrong. Editing a fixture until both sides
   pass destroys the only signal this arrangement produces, and it does so silently.

Fixtures are also the input to N20 export parity, so the same cases exercise the reference,
the exported artifact, and the port.

Some cases must be **real scenes, not synthetic arrays.** A fusion function that passes on
a uniform array proves nothing: it never sees a depth discontinuity, so it never exercises
the one case the median reduction is chosen for. At least one fixture must be a scene where
a foreground object is genuinely nearer than the surface behind it, and at least one must
be an object whose box is mostly background.
"""
