"""Tests for point-set matching metrics."""

import numpy as np

from spine.math.match import (
    intersection_size_sorted,
    overlap_chamfer,
    overlap_count,
    overlap_dice,
    overlap_iou,
    overlap_metrics,
    overlap_weighted_dice,
    overlap_weighted_iou,
)


def test_sorted_intersection_exercises_all_pointer_moves():
    """Intersection should advance either pointer and count common values."""
    assert (
        intersection_size_sorted(
            np.array([0, 2, 4, 8], dtype=np.int64),
            np.array([1, 2, 3, 8, 9], dtype=np.int64),
        )
        == 2
    )


def test_overlap_metrics_cover_empty_disjoint_and_overlapping_sets():
    """All discrete metrics should preserve empty and disjoint zero entries."""
    left = [
        np.array([], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([10], dtype=np.int64),
    ]
    right = [
        np.array([], dtype=np.int64),
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array([20], dtype=np.int64),
    ]

    count = overlap_count(left, right)
    assert count.tolist() == [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    assert overlap_iou(left, right)[1, 1] == np.float32(2 / 5)
    assert overlap_dice(left, right)[1, 1] == np.float32(4 / 7)
    assert overlap_weighted_iou(left, right)[1, 1] == np.float32(1.4)
    assert overlap_weighted_dice(left, right)[1, 1] == np.float32(2.0)


def test_combined_overlap_metrics_preserve_directionality():
    """Combined metrics should distinguish predicted purity and efficiency."""
    predicted = [
        np.array([], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([10], dtype=np.int64),
    ]
    truth = [
        np.array([], dtype=np.int64),
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array([20], dtype=np.int64),
    ]

    counts, purities, efficiencies, ious = overlap_metrics(predicted, truth)

    assert counts.tolist() == [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    np.testing.assert_array_equal(counts, overlap_count(predicted, truth))
    np.testing.assert_array_equal(ious, overlap_iou(predicted, truth))
    assert purities[1, 1] == np.float32(2 / 3)
    assert efficiencies[1, 1] == np.float32(1 / 2)
    assert ious[1, 1] == np.float32(2 / 5)
    assert not np.any(purities[[0, 2]])
    assert not np.any(efficiencies[:, [0, 2]])


def test_chamfer_distance_handles_empty_and_nonempty_clouds():
    """Chamfer matching should leave empty pairs infinite and score real pairs."""
    left = [np.empty((0, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)]
    right = [
        np.empty((0, 3), dtype=np.float32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    ]
    result = overlap_chamfer(left, right)
    assert np.isinf(result[0]).all()
    assert np.isinf(result[:, 0]).all()
    assert result[1, 1] == 2.0
