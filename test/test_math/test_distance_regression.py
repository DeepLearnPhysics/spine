"""Regression tests for distance helpers."""

import numpy as np
import pytest

from spine.math.distance import (
    METRICS,
    cdist,
    closest_pair,
    closest_pair_legacy,
    farthest_pair,
    get_metric_id,
    pdist,
)

POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    dtype=np.float32,
)


def test_get_metric_id_dispatches_minkowski_aliases_and_errors():
    """Metric dispatch should special-case p=1 and p=2 Minkowski."""
    assert get_metric_id("minkowski", 1.0) == METRICS["cityblock"]
    assert get_metric_id("minkowski", 2.0) == METRICS["euclidean"]
    assert get_metric_id("minkowski", 3.0) == METRICS["minkowski"]
    assert get_metric_id("cityblock", 2.0) == METRICS["cityblock"]
    assert get_metric_id("euclidean", 2.0) == METRICS["euclidean"]
    assert get_metric_id("sqeuclidean", 2.0) == METRICS["sqeuclidean"]
    assert get_metric_id("chebyshev", 2.0) == METRICS["chebyshev"]

    with pytest.raises(ValueError, match="not recognized"):
        get_metric_id("bad", 2.0)


def test_pdist_dispatches_all_metrics_and_errors():
    """Pairwise distance matrix should support every metric enumerator."""
    expected = {
        METRICS["minkowski"]: [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0800838],
            [2.0, 2.0800838, 0.0],
        ],
        METRICS["cityblock"]: [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        METRICS["euclidean"]: [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, np.sqrt(5.0)],
            [2.0, np.sqrt(5.0), 0.0],
        ],
        METRICS["sqeuclidean"]: [[0.0, 1.0, 4.0], [1.0, 0.0, 5.0], [4.0, 5.0, 0.0]],
        METRICS["chebyshev"]: [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
    }

    for metric, matrix in expected.items():
        np.testing.assert_allclose(pdist(POINTS, metric, p=3.0), matrix, atol=1e-5)

    with pytest.raises(ValueError, match="Distance metric"):
        pdist(POINTS, np.int64(99))


def test_cdist_dispatches_all_metrics_and_errors():
    """Cross-distance matrix should support every metric enumerator."""
    x1 = POINTS[:2]
    x2 = POINTS[1:]

    for metric in (
        METRICS["minkowski"],
        METRICS["cityblock"],
        METRICS["euclidean"],
        METRICS["sqeuclidean"],
        METRICS["chebyshev"],
    ):
        distances = cdist(x1, x2, metric, p=3.0)
        assert distances.shape == (2, 2)
        assert distances[1, 0] == 0.0

    with pytest.raises(ValueError, match="Distance metric"):
        cdist(x1, x2, np.int64(99))


def test_pair_helpers_support_iterative_paths():
    """Closest/farthest pair helpers should cover iterative variants."""
    i, j, dist = farthest_pair(POINTS, iterative=True)
    assert {i, j} == {1, 2}
    assert np.isclose(dist, np.sqrt(5.0))

    x2 = np.array([[10.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    i, j, dist = closest_pair(POINTS, x2, iterative=True, seed=False)
    assert (i, j) == (0, 1)
    assert np.isclose(dist, 0.1)

    i, j, dist = closest_pair(POINTS, x2, iterative=True, seed=True)
    assert np.isclose(dist, 0.1)


def test_legacy_closest_pair_preserves_historical_iterative_path():
    """Legacy closest pair should preserve the old one-sided iteration."""
    x1 = np.array(
        [
            [0.58366364, -1.8748202, 0.9472971],
            [-0.24740814, 0.6954392, 1.1409228],
            [0.22428122, -0.5900606, 1.20232],
        ],
        dtype=np.float32,
    )
    x2 = np.array(
        [
            [1.3192177, 0.69287896, 1.1638298],
            [-0.6025194, -0.69706947, 2.202115],
            [0.1937491, 0.1192039, 1.1976705],
            [0.3246087, -0.36247766, 1.2971592],
        ],
        dtype=np.float32,
    )

    brute = closest_pair(x1, x2, iterative=False)
    fixed = closest_pair(x1, x2, iterative=True)
    legacy = closest_pair_legacy(x1, x2, iterative=True)

    assert fixed[:2] == brute[:2] == (2, 3)
    assert np.isclose(fixed[2], brute[2])
    assert legacy[:2] == (2, 2)
    assert legacy[2] > fixed[2]


def test_legacy_closest_pair_brute_matches_current_brute():
    """Legacy closest pair should only differ on the iterative path."""
    x2 = np.array([[10.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)

    assert closest_pair_legacy(POINTS, x2, iterative=False) == closest_pair(
        POINTS, x2, iterative=False
    )


def test_farthest_pair_brute_with_non_euclidean_metric():
    """Brute farthest pair should support non-Euclidean metric branches."""
    i, j, dist = farthest_pair(POINTS, iterative=False, metric_id=METRICS["cityblock"])

    assert {i, j} == {1, 2}
    assert np.isclose(dist, 3.0)
