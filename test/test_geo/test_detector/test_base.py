"""Tests for detector geometry primitives."""

import numpy as np

from spine.geo.detector.base import Box, Plane


def test_plane_and_box_geometry():
    """Plane and box primitives should expose distances and dimensions."""
    plane = Plane(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert plane.boundary == 1.0
    assert plane.distance(np.array([3.0, 0.0, 0.0])) == 2.0

    box = Box(np.array([0.0, 0.0, 0.0]), np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(box.center, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(box.dimensions, [2.0, 4.0, 6.0])
    assert box.volume == 48.0
    assert len(box.faces) == 6
    assert box.distance(np.array([1.0, 2.0, 3.0])) == 0.0
    np.testing.assert_allclose(
        box.distance(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]])),
        [0.0, 1.0],
    )
