"""Tests for point visualization helpers."""

import numpy as np
import pytest

from spine.vis.trace.point import _prepare_point_trace_inputs, scatter_points


def test_scatter_points_supports_2d_and_rejects_bad_dimension():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    trace = scatter_points(points[:, :2], color=np.array([1.0, 2.0]))[0]

    assert trace.type == "scatter"
    assert trace.marker.color.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="dimension 2 or 3"):
        scatter_points(points, dim=4)
    with pytest.raises(ValueError, match="dimension 2 or 3"):
        _prepare_point_trace_inputs(points, dim=4)
