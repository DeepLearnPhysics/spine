"""Tests for clustering metrics."""

import numpy as np
import pytest

import spine.math.metrics as metrics_module
from spine.math.metrics import (
    _adjusted_mutual_info_score,
    _adjusted_rand_score,
    _entropy,
    ami,
    ari,
    bd,
    eff,
    pur,
    pur_eff,
    sbd,
    unique_labels,
)


def test_metrics_public_api_exposes_only_batch_aware_ari_and_ami():
    """Low-level adjusted-score implementations should remain private helpers."""
    assert "ari" in metrics_module.__all__
    assert "ami" in metrics_module.__all__
    assert "adjusted_rand_score" not in metrics_module.__all__
    assert "adjusted_mutual_info_score" not in metrics_module.__all__


def test_adjusted_rand_score_handles_perfect_random_and_empty_cases():
    """ARI should cover perfect, random-like and degenerate inputs."""
    perfect = np.array([0, 0, 1, 1], dtype=np.int32)
    crossed = np.array([0, 1, 0, 1], dtype=np.int32)
    one_cluster = np.zeros(4, dtype=np.int32)

    assert _adjusted_rand_score(perfect, perfect) == 1.0
    assert _adjusted_rand_score(crossed, perfect) <= 0.0
    assert _adjusted_rand_score(one_cluster, one_cluster) == 1.0
    assert _adjusted_rand_score(one_cluster, perfect) == 0.0
    assert np.isnan(
        _adjusted_rand_score(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
    )
    assert np.isnan(
        _adjusted_rand_score(
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        )
    )


def test_adjusted_mutual_info_score_handles_common_cases():
    """AMI should cover perfect, one-cluster and undefined empty inputs."""
    perfect = np.array([0, 0, 1, 1], dtype=np.int32)
    crossed = np.array([0, 1, 0, 1], dtype=np.int32)
    one_cluster = np.zeros(4, dtype=np.int32)

    assert _adjusted_mutual_info_score(perfect, perfect) == 1.0
    assert _adjusted_mutual_info_score(one_cluster, one_cluster) == 1.0
    assert _adjusted_mutual_info_score(perfect, one_cluster) == 0.0
    assert (
        _adjusted_mutual_info_score(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )
        == 1.0
    )
    assert _adjusted_mutual_info_score(crossed, perfect) <= 1.0
    assert (
        _adjusted_mutual_info_score(
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        )
        == 1.0
    )
    assert np.isnan(
        _adjusted_mutual_info_score(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    )


def test_entropy_handles_singleton_input():
    """Private entropy helper should handle singleton labels."""
    assert _entropy(np.array([0], dtype=np.int32)) == 0.0


def test_adjusted_mutual_info_score_rejects_length_mismatch():
    """AMI inputs must have matching lengths."""
    with pytest.raises(ValueError, match="same length"):
        _adjusted_mutual_info_score(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )


def test_adjusted_rand_score_rejects_length_mismatch():
    """ARI inputs must have matching lengths."""
    with pytest.raises(ValueError, match="same length"):
        _adjusted_rand_score(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
        )


def test_cluster_metrics_cover_empty_global_and_per_cluster_modes():
    """Public cluster metrics should support batch labels and both averaging modes."""
    empty = np.empty(0, dtype=np.int64)
    assert np.isnan(pur(empty, empty))
    assert np.isnan(eff(empty, empty))
    assert all(np.isnan(value) for value in pur_eff(empty, empty))
    assert np.isnan(ari(empty, empty))
    assert np.isnan(ami(empty, empty))
    assert np.isnan(sbd(empty, empty))
    assert np.isnan(bd(empty, empty, empty, empty, empty, empty))

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


def test_cluster_metrics_have_sensible_degenerate_partition_values():
    """All clustering scores should distinguish defined degeneracies from empties."""
    one_cluster = np.zeros(4, dtype=np.int64)
    two_clusters = np.array([0, 0, 1, 1], dtype=np.int64)

    # Identical nonempty partitions are perfect for every metric.
    assert pur(one_cluster, one_cluster) == 1.0
    assert eff(one_cluster, one_cluster) == 1.0
    assert pur_eff(one_cluster, one_cluster) == (1.0, 1.0)
    assert ari(one_cluster, one_cluster) == 1.0
    assert ami(one_cluster, one_cluster) == 1.0
    assert sbd(one_cluster, one_cluster) == 1.0

    # One cluster versus two is defined rather than an invalid sentinel.
    assert pur(one_cluster, two_clusters) == 1.0
    assert eff(one_cluster, two_clusters) == 0.5
    assert pur_eff(one_cluster, two_clusters) == (1.0, 0.5)
    assert ari(one_cluster, two_clusters) == 0.0
    assert ami(one_cluster, two_clusters) == 0.0
    assert sbd(one_cluster, two_clusters) == pytest.approx(2.0 / 3.0)

    singleton = np.zeros(1, dtype=np.int64)
    assert pur(singleton, singleton) == 1.0
    assert eff(singleton, singleton) == 1.0
    assert pur_eff(singleton, singleton) == (1.0, 1.0)
    assert np.isnan(ari(singleton, singleton))
    assert ami(singleton, singleton) == 1.0
    assert sbd(singleton, singleton) == 1.0


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
