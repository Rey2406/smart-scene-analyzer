"""The named coordinate spaces, as an enumeration rather than a convention.

**Single responsibility:** give each coordinate frame a name that can appear in a
signature, so that "which space is this box in?" is answerable by reading the type instead
of by reading the call site.

Spaces, all with origin top-left, x increasing right, y increasing down:

| Space         | Units                              | Produced by             |
|---------------|------------------------------------|-------------------------|
| `SOURCE`      | absolute pixels, decoded image     | `preprocess.image_io`   |
| `DET_INPUT`   | absolute pixels, letterboxed 640   | `preprocess.detection_input` |
| `DEPTH_INPUT` | absolute pixels, letterboxed 518   | `preprocess.depth_input`    |
| `SCREEN`      | density-independent points         | the app's overlay (`app/`)   |

`SCREEN` is declared here for completeness of the vocabulary. Nothing in this Python
package produces it — the conversion into it happens on the device, in
`app/src/geometry/toScreen.ts`. Naming it here keeps the reference and the port using one
set of words for one set of frames.

Planned public surface:

    class Space(StrEnum)                 # SOURCE | DET_INPUT | DEPTH_INPUT | SCREEN
"""
