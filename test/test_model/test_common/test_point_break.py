"""Tests for point-enhanced clustering algorithms."""

import numpy as np
import pytest

from spine.model.common.point_break import PointBreakClusterer


def test_point_break_method_dispatch_and_masked_dbscan():
    """Masked DBSCAN should split active regions and reattach masked voxels."""
    voxels = np.array([[x, 0.0, 0.0] for x in range(7)], dtype=np.float32)
    points = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
    clusterer = PointBreakClusterer(eps=1.1, mask_radius=1.1)

    labels = clusterer(voxels, points)
    assert labels.shape == (7,)
    assert len(np.unique(labels)) == 2
    np.testing.assert_array_equal(
        clusterer(voxels, points, method="masked_dbscan"), labels
    )
    with pytest.raises(ValueError, match="not recognized"):
        clusterer(voxels, points, method="unknown")


def test_point_break_masked_dbscan_all_passive_and_disconnected():
    """Groups with no active voxels and multiple preliminary groups are preserved."""
    voxels = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float32)
    points = voxels.copy()
    labels = PointBreakClusterer(eps=1.1, mask_radius=2.0).get_masked_dbscan_labels(
        voxels, points
    )
    assert len(np.unique(labels)) == 2


def test_point_break_closest_paths():
    """Closest-path clustering should construct paths through three break points."""
    voxels = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
        dtype=np.float32,
    )
    points = voxels[[0, 2, 4]]
    clusterer = PointBreakClusterer(method="closest_path", eps=1.1)
    labels = clusterer(voxels, points)
    assert labels.shape == (5,)
    assert labels.min() == 0

    # With fewer than three nearby points, the preliminary DBSCAN result stays.
    labels = clusterer.get_closest_path_labels(voxels, points[:2])
    np.testing.assert_array_equal(labels, np.zeros(5, dtype=np.int64))
