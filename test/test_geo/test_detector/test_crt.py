"""Tests for CRT detector geometry components."""

import numpy as np
import pytest

from spine.geo.detector.crt import CRTDetector, CRTPlane


def test_crt_detector_geometry():
    """CRT detector should map logical IDs and closest planes."""
    plane = CRTPlane(np.zeros(3), np.array([2.0, 4.0, 6.0]), normal_axis=2)
    np.testing.assert_array_equal(plane.normal, [0.0, 0.0, 1.0])
    assert plane.normal_axis == 2

    detector = CRTDetector(
        dimensions=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        normals=[0, 0],
        logical_ids=[5, 6],
    )
    assert len(detector) == 2
    assert detector[0] is detector.planes[0]
    assert list(detector) == detector.planes
    assert detector.num_planes == 2
    assert detector.get_plane_id(np.zeros(3), 5) == 0
    assert detector.get_plane(np.zeros(3), 6) is detector.planes[1]

    with pytest.raises(AssertionError, match="not in the detector ID mapping"):
        detector.get_plane_id(np.zeros(3), 7)


def test_crt_detector_uses_closest_plane_without_mapping():
    """CRT hits should use geometric proximity when no logical mapping exists."""
    detector = CRTDetector(
        dimensions=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        normals=[0, 0],
    )

    assert detector.get_plane_id(np.array([9.0, 0.0, 0.0]), plane_idx=999) == 1
    assert detector.get_plane(np.array([0.1, 0.0, 0.0]), plane_idx=999) is detector[0]
