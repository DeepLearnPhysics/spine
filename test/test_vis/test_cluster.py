"""Tests for cluster visualization helpers."""

import numpy as np
import pytest

from spine.vis.cluster import scatter_clusters

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
CLUSTS = [
    np.array([0, 1, 2], dtype=np.int64),
    np.array([3, 4], dtype=np.int64),
]


def test_scatter_clusters_modes_and_hover_validation():
    merged = scatter_clusters(POINTS, CLUSTS, single_trace=True)
    circles = scatter_clusters(POINTS, CLUSTS, mode="circle", color=[0, 1])
    hulls = scatter_clusters(POINTS, CLUSTS, mode="hull", color=[0.5, 1.5])

    assert len(merged) == 1
    assert len(circles) == 2
    assert len(hulls) == 2
    with pytest.raises(ValueError, match="hovertext"):
        scatter_clusters(POINTS, CLUSTS, hovertext=["a", "b", "c", "d"])


def test_scatter_clusters_color_and_mode_branches():
    per_point = scatter_clusters(POINTS, CLUSTS, color=np.arange(len(POINTS)))
    per_point_hover = scatter_clusters(
        POINTS,
        CLUSTS,
        hovertext=np.array([f"p{i}" for i in range(len(POINTS))]),
        single_trace=True,
    )
    scalar_hover = scatter_clusters(POINTS, CLUSTS, hovertext="cluster")
    scalar = scatter_clusters(POINTS, CLUSTS, color="red", shared_legend=False)
    empty = scatter_clusters(
        POINTS,
        [np.empty(0, dtype=np.int64)],
        single_trace=True,
    )
    circle = scatter_clusters(POINTS, CLUSTS, mode="circle", single_trace=True)
    ellipsoid = scatter_clusters(
        POINTS,
        CLUSTS,
        mode="ellipsoid",
        color=[0.0, 1.0],
        name=["a", "b"],
        shared_legend=False,
    )
    cone = scatter_clusters(
        POINTS,
        CLUSTS,
        mode="cone",
        color=[0.0, 1.0],
        shared_legend=False,
    )

    assert len(per_point) == 2
    assert len(per_point_hover[0].x) == len(POINTS)
    assert "cluster" in scalar_hover[0].hovertemplate
    assert len(empty[0].x) == 0
    assert per_point[0].marker.color.tolist() == [0, 1, 2]
    assert scalar[1].marker.color == "red"
    assert len(circle) == 1
    assert ellipsoid[0].name == "a"
    assert len(cone) == 2


def test_cluster_validation_rejects_inconsistent_arrays():
    with pytest.raises(ValueError, match="color"):
        scatter_clusters(POINTS, CLUSTS, color=np.arange(3))
    with pytest.raises(ValueError, match="hovertext"):
        scatter_clusters(POINTS, CLUSTS, hovertext=np.arange(3))
    with pytest.raises(ValueError, match="Can only combine"):
        scatter_clusters(POINTS, CLUSTS, single_trace=True, mode="hull")
    with pytest.raises(ValueError, match="Cannot split legend"):
        scatter_clusters(POINTS, CLUSTS, single_trace=True, shared_legend=False)
    with pytest.raises(ValueError, match="one name per cluster"):
        scatter_clusters(POINTS, CLUSTS, name=["a"], shared_legend=False)
    with pytest.raises(ValueError, match="not recognized"):
        scatter_clusters(POINTS, CLUSTS, mode="bad")
