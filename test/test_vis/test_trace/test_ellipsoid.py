"""Tests for ellipsoid visualization helpers."""

import numpy as np
import pytest

from spine.vis.trace.ellipsoid import ellipsoid_trace, ellipsoid_traces

POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


def test_ellipsoid_helpers_build_from_points_and_covariance():
    ell_from_points = ellipsoid_trace(points=POINTS, color="green")
    ell_from_one_point = ellipsoid_trace(points=POINTS[:1])
    ell_from_cov = ellipsoid_trace(centroid=np.zeros(3), covmat=np.eye(3))
    ell_with_text = ellipsoid_trace(
        centroid=np.zeros(3),
        covmat=np.eye(3),
        hovertext=["a"] * 100,
    )
    ell_list = ellipsoid_traces(
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.eye(3),
        color=np.array([1.0, 2.0]),
        hovertext=np.array(["a", "b"]),
        shared_legend=False,
    )
    auto_hover = ellipsoid_traces(
        np.array([[0, 0, 0]], dtype=np.float32),
        np.eye(3),
        color=np.array([1.0]),
    )

    assert ell_from_points.color == "green"
    assert np.all(np.isfinite(ell_from_one_point.x))
    assert np.all(np.isfinite(ell_from_cov.x))
    assert ell_with_text.hovertext == ("a",) * 100
    assert len(ell_list) == 2
    assert "Value:" in auto_hover[0].hovertemplate


def test_ellipsoid_validation_paths():
    centroids = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

    with pytest.raises(ValueError, match="either `points`"):
        ellipsoid_trace()
    with pytest.raises(ValueError, match="\\(N, 3\\) array"):
        ellipsoid_trace(points=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="both `color` and `intensity`"):
        ellipsoid_trace(
            centroid=np.zeros(3),
            covmat=np.eye(3),
            color=1.0,
            intensity=np.ones(4),
        )
    with pytest.raises(ValueError, match="probability"):
        ellipsoid_trace(points=POINTS, contour=1.5)
    with pytest.raises(ValueError, match="one color"):
        ellipsoid_traces(centroids, np.eye(3), color=np.arange(3))
    with pytest.raises(ValueError, match="one hovertext"):
        ellipsoid_traces(centroids, np.eye(3), hovertext=np.arange(3))
