"""Tests for optical detector geometry components."""

import numpy as np
import pytest

from spine.geo.detector.optical import OptDetector, OpticalVolume


def test_optical_volume_and_detector_properties():
    """Optical detector geometry should expose volume and channel metadata."""
    volume = OpticalVolume(
        centroid=np.zeros(3),
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        sizes=np.array([[1.0, 1.0, 1.0]]),
        shape="box",
    )
    assert volume.num_detectors == 2
    np.testing.assert_allclose(volume.lower, [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(volume.upper, [2.5, 0.5, 0.5])

    detector = OptDetector(
        volume="tpc",
        volume_offsets=[np.zeros(3), np.array([10.0, 0.0, 0.0])],
        shape=["box", "disk"],
        dimensions=[[1.0, 1.0, 1.0], [2.0, 2.0, 0.5]],
        positions=[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        shape_ids=[0, 1],
        det_ids=[0, 1],
        mirror=True,
    )

    assert detector.num_volumes == 2
    assert detector.num_detectors_per_volume == 2
    assert detector.num_detectors == 4
    assert detector.num_channels == 4
    assert detector.num_channels_per_volume == 2
    np.testing.assert_array_equal(detector.volume_index(1), [2, 3])
    assert detector.positions.shape == (4, 3)
    np.testing.assert_array_equal(detector.sizes, [[1.0, 1.0, 1.0], [2.0, 2.0, 0.5]])
    assert detector.shape == ["box", "disk"]
    np.testing.assert_array_equal(detector.shape_ids, [0, 1, 0, 1])
    np.testing.assert_array_equal(detector.det_ids, [0, 1, 2, 3])


def test_optical_single_shape_without_channel_map():
    """Single-shape detectors without det_ids should use detector counts."""
    detector = OptDetector(
        volume="module",
        volume_offsets=[np.zeros(3), np.array([10.0, 0.0, 0.0])],
        shape="box",
        dimensions=[1.0, 1.0, 1.0],
        positions=[[0.0, 0.0, 0.0]],
    )

    assert detector.num_channels == detector.num_detectors
    assert detector.num_channels_per_volume == detector.num_detectors_per_volume
    assert detector.shape_ids is None
    assert detector.det_ids is None


def test_optical_single_volume_channel_map():
    """Single-volume det_ids should be returned without volume offsets."""
    detector = OptDetector(
        volume="module",
        volume_offsets=[np.zeros(3)],
        shape="box",
        dimensions=[1.0, 1.0, 1.0],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        det_ids=[0, 1],
    )

    np.testing.assert_array_equal(detector.det_ids, [0, 1])


def test_optical_global_index_and_validation():
    """OptDetector should support global indexes and reject invalid configs."""
    detector = OptDetector(
        volume="module",
        volume_offsets=[np.zeros(3), np.array([10.0, 0.0, 0.0])],
        shape="box",
        dimensions=[1.0, 1.0, 1.0],
        positions=[[0.0, 0.0, 0.0]],
        global_index=True,
    )
    np.testing.assert_array_equal(detector.volume_index(1), [0, 1])

    with pytest.raises(AssertionError, match="segmentation"):
        OptDetector("bad", [np.zeros(3)], "box", [1, 1, 1], [[0, 0, 0]])

    with pytest.raises(AssertionError, match="shape map"):
        OptDetector("tpc", [np.zeros(3)], ["box", "disk"], [[1, 1, 1]], [[0, 0, 0]])
