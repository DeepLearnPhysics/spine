"""Tests for predicted-cluster overlap quality measurements."""

import numpy as np

from spine.cluster import get_cluster_overlap_batch
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch


def test_cluster_overlap_batch_reports_best_match_quality():
    """Best-match metrics should preserve batch boundaries and direction."""
    rows = np.zeros((6, 7), dtype=np.float32)
    rows[:, 0] = (0, 0, 0, 0, 0, 1)
    rows[:, 1] = np.arange(6)
    rows[:, 4] = 1.0
    rows[:, 5] = (0, 0, 0, 1, 1, 0)
    rows[:, 6] = (0, 0, 0, 1, 1, 0)
    particles = {
        "group": TensorBatch(np.array([0, 1, -1]), counts=[2, 1]),
    }
    labels = ClusterLabelBatch(
        TensorBatch(rows, counts=[5, 1], has_batch_col=True),
        particles,
    )
    objects = IndexBatch(
        [
            np.array([0, 1]),
            np.array([2, 3]),
            np.array([4]),
            np.array([5]),
        ],
        spans=[5, 1],
        counts=[3, 1],
        single_counts=[2, 2, 1, 1],
    )

    overlap = get_cluster_overlap_batch(labels, objects)

    np.testing.assert_array_equal(overlap.match_ids.data, [0, 0, 1, -1])
    np.testing.assert_array_equal(overlap.intersections.data, [2, 1, 1, 0])
    np.testing.assert_allclose(overlap.purities.data, [1.0, 0.5, 1.0, 0.0])
    np.testing.assert_allclose(
        overlap.efficiencies.data,
        [2 / 3, 1 / 3, 1 / 2, 0.0],
    )
    np.testing.assert_allclose(overlap.ious.data, [2 / 3, 1 / 4, 1 / 2, 0.0])


def test_cluster_overlap_batch_rejects_misaligned_spans():
    """Overlap quality requires indexes in the cluster-label voxel space."""
    rows = np.zeros((2, 7), dtype=np.float32)
    labels = ClusterLabelBatch(
        TensorBatch(rows, counts=[2], has_batch_col=True),
        {"group": TensorBatch(np.array([0]), counts=[1])},
    )
    objects = IndexBatch(
        [np.array([0])],
        spans=[1],
        counts=[1],
        single_counts=[1],
    )

    with np.testing.assert_raises_regex(ValueError, "share batch spans"):
        get_cluster_overlap_batch(labels, objects)


def test_cluster_overlap_batch_accepts_empty_event():
    """Events without predicted clusters should preserve batch alignment."""
    rows = np.zeros((2, 7), dtype=np.float32)
    rows[:, 0] = (0, 1)
    rows[:, 5] = (0, 0)
    labels = ClusterLabelBatch(
        TensorBatch(rows, counts=[1, 1], has_batch_col=True),
        {"group": TensorBatch(np.array([0, 0]), counts=[1, 1])},
    )
    objects = IndexBatch(
        [np.array([1])],
        spans=[1, 1],
        counts=[0, 1],
        single_counts=[1],
    )

    overlap = get_cluster_overlap_batch(labels, objects)

    np.testing.assert_array_equal(overlap.match_ids.data, [0])
    np.testing.assert_allclose(overlap.ious.data, [1.0])
