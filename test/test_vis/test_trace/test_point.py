"""Tests for point visualization helpers."""

import numpy as np
import pytest

from spine.data import TensorBatch, TensorData
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


def test_scatter_points_accepts_typed_products_and_validates_coordinates():
    """Point traces should consume typed products and require coordinate data."""
    points = np.arange(9, dtype=np.float32).reshape(3, 3)
    point_data = TensorData(np.ones((3, 1)), coords=points)
    point_batch = TensorBatch.from_data_list([point_data])

    data_trace = scatter_points(point_data)[0]
    batch_trace = scatter_points(point_batch)[0]
    _, _, _, shared_hovertext, shared_template = _prepare_point_trace_inputs(
        points, hovertext="shared context"
    )

    np.testing.assert_array_equal(data_trace.x, points[:, 0])
    np.testing.assert_array_equal(batch_trace.z, points[:, 2])
    assert shared_hovertext is None
    assert "shared context" in shared_template
    with pytest.raises(ValueError, match="does not carry coordinates"):
        scatter_points(TensorData(np.ones((3, 1)), feats_only=True))
    with pytest.raises(ValueError, match="coordinates only"):
        scatter_points(np.ones((3, 4)))
