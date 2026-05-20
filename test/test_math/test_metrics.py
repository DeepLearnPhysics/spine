"""Tests for clustering metrics."""

import numpy as np
import pytest

from spine.math.metrics import _entropy, adjusted_mutual_info_score, adjusted_rand_score


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
