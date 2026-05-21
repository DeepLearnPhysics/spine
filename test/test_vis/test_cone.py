"""Tests for cone visualization helpers."""

import numpy as np
import pytest

from spine.vis.cone import cone_trace

POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def test_cone_trace_uses_string_color_as_mesh_color():
    trace = cone_trace(POINTS, color="red")

    assert trace.color == "red"
    assert trace.intensity is None


def test_cone_trace_forwards_hovertext_and_validates_inputs():
    trace = cone_trace(POINTS, hovertext=["cone"] * 100)

    assert trace.hovertext == ("cone",) * 100
    with pytest.raises(ValueError, match="probability"):
        cone_trace(POINTS, fraction=1.5)
    with pytest.raises(ValueError, match="either `color` or `intensity`"):
        cone_trace(POINTS, color=1.0, intensity=np.ones(10))
    with pytest.raises(ValueError, match="single color"):
        cone_trace(POINTS, color=np.array([1.0, 2.0]))
