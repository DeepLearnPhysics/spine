"""Tests for visualization primitive helpers."""

import numpy as np
import pytest

from spine.vis.cone import cone_trace
from spine.vis.cylinder import cylinder_trace
from spine.vis.lite import em_cone_trace
from spine.vis.utils import rotation_matrix_from_z


def test_rotation_matrix_from_z_handles_antiparallel_direction():
    """A direction opposite to z should not trigger Rodrigues division by zero."""
    rotmat = rotation_matrix_from_z(np.array([0.0, 0.0, -2.0]))

    np.testing.assert_allclose(rotmat @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])
    np.testing.assert_allclose(rotmat.T @ rotmat, np.eye(3))


def test_rotation_matrix_from_z_rejects_zero_direction():
    """A zero direction cannot define a rotation target."""
    with pytest.raises(ValueError, match="zero direction"):
        rotation_matrix_from_z(np.zeros(3))


def test_cylinder_trace_handles_antiparallel_axis():
    """Cylinder rendering should be finite for axes opposite to z."""
    trace = cylinder_trace(
        centroid=np.zeros(3),
        axis=np.array([0.0, 0.0, -1.0]),
        height=4.0,
        diameter=2.0,
    )

    assert np.all(np.isfinite(trace.x))
    assert np.all(np.isfinite(trace.y))
    assert np.all(np.isfinite(trace.z))
    assert np.isclose(np.min(trace.z), -2.0)
    assert np.isclose(np.max(trace.z), 2.0)


def test_cone_trace_uses_string_color_as_mesh_color():
    """A string cone color should be passed as Mesh3d color, not intensity."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    trace = cone_trace(points, color="red")

    assert trace.color == "red"
    assert trace.intensity is None


def test_em_cone_trace_handles_antiparallel_direction_and_color():
    """Lite cone rendering should support -z directions and string colors."""
    trace = em_cone_trace(
        start_point=np.zeros(3),
        direction=np.array([0.0, 0.0, -1.0]),
        energy=100.0,
        color="blue",
    )

    assert trace.color == "blue"
    assert trace.intensity is None
    assert np.all(np.isfinite(trace.x))
    assert np.all(np.isfinite(trace.y))
    assert np.all(np.isfinite(trace.z))
    assert np.max(trace.z) <= 1.0e-8
