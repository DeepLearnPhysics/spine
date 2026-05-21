"""Tests for hull visualization helpers."""

import numpy as np
import pytest

from spine.vis.hull import hull_trace

POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


def test_hull_trace_supports_numeric_color_and_hovertext():
    hull = hull_trace(POINTS, color=3.0)
    hull_text = hull_trace(POINTS, hovertext=["a"] * len(POINTS))

    assert hull.intensity.tolist() == [3.0] * len(POINTS)
    assert hull_text.hovertext == ("a",) * len(POINTS)


def test_hull_trace_rejects_color_and_intensity_conflict():
    with pytest.raises(ValueError, match="both `color` and `intensity`"):
        hull_trace(POINTS, color=1.0, intensity=np.ones(len(POINTS)))
