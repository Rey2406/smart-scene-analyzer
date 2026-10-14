"""Offline tests for the depth module.

No GPU, no network, no weights on disk. `depth.load_model` takes a `loader` seam for
exactly this reason, so every test here substitutes a fake and nothing touches
`transformers` or the Hugging Face cache.

The key fixture is a depth map with a **known gradient**: every expected median can be
computed by hand, so assertions are exact rather than "seems plausible".
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from smart_scene_analyzer import depth

BANNED_SUBSTRINGS = ("meter", "metre", "_mm", "_cm", "distance")


# ------------------------------------------------------------------------- fixtures


@pytest.fixture
def known_gradient() -> np.ndarray:
    """20x10 float32 whose value equals its column index.

    Column index is the value, so the median of any box is the median of the column
    indices it spans — computable on paper.
    """
    h, w = 20, 10
    return np.tile(np.arange(w, dtype=np.float32), (h, 1))


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), mode="RGB")


@pytest.fixture
def sample_image_nonsquare() -> Image.Image:
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (13, 41, 3), dtype=np.uint8), mode="RGB")


def fake_loader(raster: np.ndarray):
    """A loader returning a fixed raster, ignoring the image. No weights, no network."""

    def build(_device: str):
        def run(_image):
            return raster

        return run

    return build


# ------------------------------------------------- the test worth arguing about

def test_region_actually_uses_the_depth_map(known_gradient):
    """A reducer that ignored its map would pass every other test in this file.

    Returning a constant satisfies the shape assertions, the degenerate-case
    assertions, and even an ordering assertion on a single fixture. It cannot
    survive the same box over two different maps.
    """
    box = (2.0, 0.0, 6.0, 5.0)
    first = depth.region_relative_depth(known_gradient, box)
    second = depth.region_relative_depth(known_gradient * -1.0 + known_gradient.max(), box)
    assert first != second


def test_estimate_actually_uses_the_model_output(sample_image):
    """Same question one level up: does the public function pass the model's raster through?"""
    low = depth.estimate_relative_inverse_depth(
        sample_image, loader=fake_loader(np.full((24, 32), 1.0, dtype=np.float32))
    )
    high = depth.estimate_relative_inverse_depth(
        sample_image, loader=fake_loader(np.full((24, 32), 9.0, dtype=np.float32))
    )
    assert not np.allclose(low, high)


# ------------------------------------------------------------- exact values on a gradient


def test_median_of_a_box_is_exact(known_gradient):
    # columns 2,3,4,5 -> median of [2,3,4,5] = 3.5
    assert depth.region_relative_depth(known_gradient, (2, 0, 6, 5)) == pytest.approx(3.5)


def test_median_not_mean(known_gradient):
    """Separate the two statistics, which a linear gradient alone cannot do.

    On an evenly-spaced region median and mean are identical, so a box on the raw
    gradient would pass against either implementation. One outlier column — an
    occluding surface across a corner, which is the real case the docstring argues
    about — pulls the mean and leaves the median where it was.
    """
    skewed = known_gradient.copy()
    skewed[:, 3] = 1000.0  # one column far nearer than the rest
    value = depth.region_relative_depth(skewed, (0, 0, 4, 5))  # columns 0,1,2,3
    assert value == pytest.approx(1.5), "median of [0,1,2,1000] is 1.5"
    mean_would_be = float(np.mean(skewed[0:5, 0:4]))
    assert mean_would_be > 200.0, "the mean is dragged by the outlier; the median is not"


def test_larger_is_nearer_is_preserved_by_the_reducer(known_gradient):
    """The sign convention is the module's central claim; assert it on the contract."""
    near = depth.region_relative_depth(known_gradient, (8, 0, 10, 5))  # high columns
    far = depth.region_relative_depth(known_gradient, (0, 0, 2, 5))  # low columns
    assert near > far


# ------------------------------------------------------------------ degenerate box cases


@pytest.mark.parametrize(
    "box",
    [(5, 5, 5, 8), (5, 5, 8, 5), (6, 6, 2, 9)],  # zero width, zero height, inverted
)
def test_zero_area_box_raises(known_gradient, box):
    with pytest.raises(depth.DegenerateRegionError, match="zero-area"):
        depth.region_relative_depth(known_gradient, box)


def test_box_partly_outside_is_clipped_not_an_error(known_gradient):
    """The common case at frame edges. Clipping to columns 0..3 gives median 1.5."""
    assert depth.region_relative_depth(known_gradient, (-50, -50, 4, 5)) == pytest.approx(1.5)


@pytest.mark.parametrize("box", [(100, 0, 150, 5), (-80, 0, -10, 5), (0, 90, 5, 120)])
def test_box_entirely_outside_raises(known_gradient, box):
    with pytest.raises(depth.DegenerateRegionError, match="encloses no pixel"):
        depth.region_relative_depth(known_gradient, box)


def test_degenerate_result_is_never_a_quiet_nan(known_gradient):
    """The documented contract: raise, so a NaN cannot propagate into a caller."""
    with pytest.raises(depth.DegenerateRegionError):
        depth.region_relative_depth(known_gradient, (200, 200, 260, 260))


def test_non_2d_map_raises(known_gradient):
    with pytest.raises(ValueError, match="HxW"):
        depth.region_relative_depth(known_gradient[None, ...], (0, 0, 2, 2))


# ----------------------------------------------------------------------- shapes and spaces


def test_output_is_on_the_source_grid_not_the_model_raster(sample_image):
    """The model raster is a different size; the public function must resize to source."""
    model_raster = np.linspace(0, 1, 14 * 28, dtype=np.float32).reshape(14, 28)
    out = depth.estimate_relative_inverse_depth(sample_image, loader=fake_loader(model_raster))
    assert out.shape == (sample_image.height, sample_image.width)
    assert out.dtype == np.float32


def test_raw_raster_is_reachable_and_unresized(sample_image):
    """Lesson 04 compares this grid against the export, so it must not be resized."""
    model_raster = np.linspace(0, 1, 14 * 28, dtype=np.float32).reshape(14, 28)
    raw = depth.estimate_raw_raster(sample_image, loader=fake_loader(model_raster))
    assert raw.shape == (14, 28)


def test_nonsquare_source_keeps_its_aspect(sample_image_nonsquare):
    model_raster = np.zeros((14, 14), dtype=np.float32)
    out = depth.estimate_relative_inverse_depth(
        sample_image_nonsquare, loader=fake_loader(model_raster)
    )
    assert out.shape == (sample_image_nonsquare.height, sample_image_nonsquare.width)


def test_resize_is_identity_when_shapes_already_match(known_gradient):
    out = depth.resize_raster_to_source(known_gradient, known_gradient.shape)
    assert np.array_equal(out, known_gradient)


def test_accepts_numpy_array_as_well_as_pil():
    arr = np.zeros((16, 20, 3), dtype=np.uint8)
    out = depth.estimate_relative_inverse_depth(
        arr, loader=fake_loader(np.zeros((14, 14), dtype=np.float32))
    )
    assert out.shape == (16, 20)


def test_rejects_an_image_that_is_not_hxwx3():
    with pytest.raises(ValueError, match="HxWx3"):
        depth.estimate_relative_inverse_depth(
            np.zeros((16, 20), dtype=np.uint8),
            loader=fake_loader(np.zeros((14, 14), dtype=np.float32)),
        )


def test_model_returning_a_non_2d_raster_raises(sample_image):
    with pytest.raises(ValueError, match="expected HxW"):
        depth.estimate_raw_raster(
            sample_image, loader=fake_loader(np.zeros((2, 14, 14), dtype=np.float32))
        )


# --------------------------------------------------------------------- naming contract


def test_no_public_name_implies_a_metric_unit():
    """Walk the module programmatically rather than hardcoding today's names.

    `units_guard.py` blocks this at write time; this asserts it at test time, so a
    rename that slipped through a path the hook does not watch still fails here.
    """
    offenders = [
        name
        for name in dir(depth)
        if not name.startswith("_")
        and any(bad in name.lower() for bad in BANNED_SUBSTRINGS)
    ]
    assert offenders == []


def test_the_public_functions_carry_the_units_in_their_names():
    assert hasattr(depth, "estimate_relative_inverse_depth")
    assert hasattr(depth, "region_relative_depth")


def test_docstrings_state_the_four_claims():
    """Sign convention, arbitrary scale, not a unit, and no ground truth."""
    text = (depth.__doc__ or "") + (depth.estimate_relative_inverse_depth.__doc__ or "")
    lowered = text.lower()
    assert "larger" in lowered and "nearer" in lowered
    assert "arbitrary" in lowered
    assert "not a distance, in any unit" in lowered
    assert "ground truth" in lowered


def test_module_does_not_import_transformers_at_module_scope():
    """Loading weights at import would make the whole suite need the network.

    Asserted against the source above `_build_default_model`, because `sys.modules` is
    useless here — another test in the session may legitimately have imported
    `transformers`, so checking it would pass or fail on test ordering rather than on
    this module's behaviour.
    """
    import inspect

    header = inspect.getsource(depth).split("def _build_default_model")[0]
    assert "import transformers" not in header
    assert "from transformers" not in header
    # ...and the real import is where it belongs: inside the builder.
    builder = inspect.getsource(depth._build_default_model)
    assert "from transformers import" in builder
