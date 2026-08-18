"""Tests for clustering metrics."""

import numpy as np
import pytest

from spine.math.metrics import (
    _entropy,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    ami,
    ari,
    bd,
    eff,
    pur,
    pur_eff,
    sbd,
    unique_labels,
)


def test_adjusted_rand_score_handles_perfect_random_and_empty_cases():
    """ARI should cover perfect, random-like and degenerate inputs."""
    perfect = np.array([0, 0, 1, 1], dtype=np.int32)
    crossed = np.array([0, 1, 0, 1], dtype=np.int32)
    one_cluster = np.zeros(4, dtype=np.int32)

    assert adjusted_rand_score(perfect, perfect) == 1.0
    assert adjusted_rand_score(crossed, perfect) <= 0.0
    assert adjusted_rand_score(one_cluster, one_cluster) == 1.0
    assert (
        adjusted_rand_score(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
        == 1.0
    )


def test_adjusted_mutual_info_score_handles_common_cases():
    """AMI should cover perfect, one-cluster and empty inputs."""
    perfect = np.array([0, 0, 1, 1], dtype=np.int32)
    crossed = np.array([0, 1, 0, 1], dtype=np.int32)
    one_cluster = np.zeros(4, dtype=np.int32)

    assert adjusted_mutual_info_score(perfect, perfect) == 1.0
    assert adjusted_mutual_info_score(one_cluster, one_cluster) == 1.0
    assert adjusted_mutual_info_score(perfect, one_cluster) == 0.0
    assert (
        adjusted_mutual_info_score(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )
        == 1.0
    )
    assert adjusted_mutual_info_score(crossed, perfect) <= 1.0
    assert (
        adjusted_mutual_info_score(
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        )
        == 1.0
    )
    assert (
        adjusted_mutual_info_score(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
        == 1.0
    )


def test_entropy_handles_singleton_input():
    """Private entropy helper should handle singleton labels."""
    assert _entropy(np.array([0], dtype=np.int32)) == 0.0


def test_adjusted_mutual_info_score_rejects_length_mismatch():
    """AMI inputs must have matching lengths."""
    with pytest.raises(ValueError, match="same length"):
        adjusted_mutual_info_score(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )


def test_adjusted_rand_score_rejects_length_mismatch():
    """ARI inputs must have matching lengths."""
    with pytest.raises(ValueError, match="same length"):
        adjusted_rand_score(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )


def test_cluster_metrics_cover_empty_global_and_per_cluster_modes():
    """Public cluster metrics should support batch labels and both averaging modes."""
    empty = np.empty(0, dtype=np.int64)
    assert pur(empty, empty) == -1.0
    assert eff(empty, empty) == -1.0
    assert pur_eff(empty, empty) == (-1.0, -1.0)
    assert ari(empty, empty) == -1.0
    assert ami(empty, empty) == -1.0
    assert bd(empty, empty, empty, empty, empty, empty) == -1.0

    truth = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    pred = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)
    batches = np.repeat([0, 1], 4)
    for per_cluster in (True, False):
        purity = pur(truth, pred, batches, per_cluster)
        efficiency = eff(truth, pred, batches, per_cluster)
        combined = pur_eff(truth, pred, batches, per_cluster)
        assert combined == pytest.approx((purity, efficiency))
        assert 0.0 <= purity <= 1.0
        assert 0.0 <= efficiency <= 1.0

    assert -1.0 <= ari(truth, pred, batches) <= 1.0
    assert -1.0 <= ami(truth, pred, batches) <= 1.0
    assert 0.0 <= sbd(truth, pred, batches) <= 1.0


def test_unique_labels_and_best_dice_known_partition():
    """Batch-aware relabeling and best Dice should retain cluster multiplicities."""
    labels = np.array([4, 4, 4, 4], dtype=np.int64)
    batches = np.array([0, 0, 1, 1], dtype=np.int64)
    inverse, unique, counts = unique_labels(labels, batches)
    assert inverse.tolist() == [0, 0, 1, 1]
    assert unique.shape == (2, 2)
    assert counts.tolist() == [2, 2]

    inverse, unique, counts = unique_labels(labels)
    assert inverse.tolist() == [0, 0, 0, 0]
    assert unique.tolist() == [4]
    assert counts.tolist() == [4]

    truth = np.array([0, 0, 1, 1], dtype=np.int64)
    pred = np.array([0, 0, 0, 1], dtype=np.int64)
    truth_values, truth_counts = np.unique(truth, return_counts=True)
    pred_values, pred_counts = np.unique(pred, return_counts=True)
    assert bd(
        truth,
        truth_values,
        truth_counts,
        pred,
        pred_values,
        pred_counts,
    ) == pytest.approx((0.8 + 2.0 / 3.0) / 2.0)
