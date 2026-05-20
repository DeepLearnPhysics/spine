"""Validation tests for visualization helper inputs."""

import numpy as np
import pytest

from spine.vis.arrow import scatter_arrows
from spine.vis.box import box_trace
from spine.vis.cluster import scatter_clusters
from spine.vis.cone import cone_trace
from spine.vis.cylinder import cylinder_trace, cylinder_traces
from spine.vis.ellipsoid import ellipsoid_trace, ellipsoid_traces
from spine.vis.layout import layout3d
from spine.vis.lite import scatter_lite_interactions, scatter_lite_particles
from spine.vis.metric.confmat import build_matrix, draw_confusion_matrix, rebuild_matrix
from spine.vis.network import network_schematic, network_topology

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
    np.array([0, 1], dtype=np.int64),
    np.array([2, 3], dtype=np.int64),
]


def test_arrow_validation_rejects_mismatched_colors():
    """Arrow colors should be scalar or match the number of arrows."""
    with pytest.raises(ValueError, match="length must match"):
        scatter_arrows(
            POINTS[:2],
            np.ones((2, 3), dtype=np.float32),
            color=np.array([1.0, 2.0, 3.0]),
        )


def test_box_validation_rejects_bad_bounds_and_color_conflicts():
    """Box helpers should raise explicit exceptions for invalid inputs."""
    with pytest.raises(ValueError, match="3 values"):
        box_trace(np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="greater"):
        box_trace(np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="Must not specify `line`"):
        box_trace(np.zeros(3), np.ones(3), line={}, color="red")
    with pytest.raises(ValueError, match="both `color` and `intensity`"):
        box_trace(
            np.zeros(3),
            np.ones(3),
            draw_faces=True,
            color=1.0,
            intensity=np.ones(8),
        )
    with pytest.raises(ValueError, match="upper boundary"):
        from spine.vis.box import box_traces

        box_traces(np.zeros((2, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="one color"):
        from spine.vis.box import box_traces

        box_traces(np.zeros((1, 3)), np.ones((1, 3)), color=np.arange(2))
    with pytest.raises(ValueError, match="one hovertext"):
        from spine.vis.box import box_traces

        box_traces(np.zeros((1, 3)), np.ones((1, 3)), hovertext=np.arange(2))
    with pytest.raises(ValueError, match="three dimensions"):
        from spine.vis.box import scatter_boxes

        scatter_boxes(np.zeros((1, 3)), dimension=np.ones(2))


def test_cluster_validation_rejects_inconsistent_arrays():
    """Cluster drawing should validate color, hovertext, and legend requests."""
    with pytest.raises(ValueError, match="color"):
        scatter_clusters(POINTS, CLUSTS, color=np.arange(3))
    with pytest.raises(ValueError, match="hovertext"):
        scatter_clusters(POINTS, CLUSTS, hovertext=np.arange(3))
    with pytest.raises(ValueError, match="Can only combine"):
        scatter_clusters(POINTS, CLUSTS, single_trace=True, mode="hull")
    with pytest.raises(ValueError, match="Cannot split legend"):
        scatter_clusters(
            POINTS,
            CLUSTS,
            single_trace=True,
            shared_legend=False,
        )
    with pytest.raises(ValueError, match="one name per cluster"):
        scatter_clusters(POINTS, CLUSTS, name=["a"], shared_legend=False)


def test_cone_cylinder_and_ellipsoid_validation_paths():
    """Mesh helpers should validate probability and conflicting color arguments."""
    with pytest.raises(ValueError, match="probability"):
        cone_trace(POINTS, fraction=1.5)
    with pytest.raises(ValueError, match="either `color` or `intensity`"):
        cone_trace(POINTS, color=1.0, intensity=np.ones(10))
    with pytest.raises(ValueError, match="single color"):
        cone_trace(POINTS, color=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="both `color` and `intensity`"):
        cylinder_trace(
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            1.0,
            1.0,
            color=1.0,
            intensity=np.ones(4),
        )
    with pytest.raises(ValueError, match="both `color` and `intensity`"):
        ellipsoid_trace(
            centroid=np.zeros(3),
            covmat=np.eye(3),
            color=1.0,
            intensity=np.ones(4),
        )
    with pytest.raises(ValueError, match="probability"):
        ellipsoid_trace(points=POINTS, contour=1.5)


def test_cylinder_and_ellipsoid_list_validation_paths():
    """Vectorized mesh helpers should validate per-object array lengths."""
    centroids = POINTS[:2]
    with pytest.raises(ValueError, match="one color"):
        cylinder_traces(
            centroids,
            np.array([0.0, 0.0, 1.0]),
            1.0,
            1.0,
            color=np.arange(3),
        )
    with pytest.raises(ValueError, match="one hovertext"):
        cylinder_traces(
            centroids,
            np.array([0.0, 0.0, 1.0]),
            1.0,
            1.0,
            hovertext=np.arange(3),
        )
    with pytest.raises(ValueError, match="one axis"):
        cylinder_traces(centroids, np.ones((3, 3)), 1.0, 1.0)
    with pytest.raises(ValueError, match="one height"):
        cylinder_traces(
            centroids,
            np.array([0.0, 0.0, 1.0]),
            np.arange(3),
            1.0,
        )
    with pytest.raises(ValueError, match="one diameter"):
        cylinder_traces(
            centroids,
            np.array([0.0, 0.0, 1.0]),
            1.0,
            np.arange(3),
        )
    with pytest.raises(ValueError, match="one color"):
        ellipsoid_traces(centroids, np.eye(3), color=np.arange(3))
    with pytest.raises(ValueError, match="one hovertext"):
        ellipsoid_traces(centroids, np.eye(3), hovertext=np.arange(3))


def test_network_validation_rejects_ambiguous_color_and_labels():
    """Network helpers should reject ambiguous color input and bad labels."""
    edges = np.array([[0, 1]], dtype=np.int64)
    labels = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="clust_labels"):
        network_topology(POINTS, CLUSTS, edges, color="red")
    with pytest.raises(ValueError, match="clust_labels"):
        network_schematic(CLUSTS, edges, labels, color="red")
    with pytest.raises(ValueError, match="primary label"):
        network_schematic(CLUSTS, edges, np.array([0]))
    with pytest.raises(ValueError, match="0 or 1"):
        network_schematic(CLUSTS, edges, np.array([0, 2]))


def test_confusion_matrix_validation_paths(tmp_path):
    """Confusion-matrix helpers should reject inconsistent configuration."""
    import pandas as pd

    data = pd.DataFrame(
        {
            "pred": [0, 1],
            "label": [0, 1],
            "score_0": [0.8, 0.1],
            "score_1": [0.2, 0.9],
        }
    )
    matrix_data = pd.DataFrame(
        {
            "count_00": [1],
            "count_01": [2],
            "count_10": [3],
            "count_11": [4],
        }
    )
    path = tmp_path / "confmat.csv"
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="number of classes"):
        build_matrix(data, num_classes=3, mapping={0: [0], 1: [1]})
    with pytest.raises(ValueError, match="number of classes"):
        rebuild_matrix(matrix_data, num_classes=3, mapping={0: [0], 1: [1]})
    with pytest.raises(ValueError, match="normalization axis"):
        draw_confusion_matrix(path, num_classes=2, norm_axis=2)
    with pytest.raises(ValueError, match="one class label"):
        draw_confusion_matrix(path, num_classes=2, class_names=["a"])


def test_layout_validation_paths():
    """3D layout configuration should reject ambiguous range inputs."""
    bad_shape = np.ones((2, 2), dtype=np.float32)
    bad_bounds = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="shape"):
        layout3d(ranges=bad_shape)
    with pytest.raises(ValueError, match="upper bound"):
        layout3d(ranges=bad_bounds)
    with pytest.raises(ValueError, match="one title"):
        layout3d(titles=["x"])


def test_lite_validation_rejects_mismatched_colors():
    """Lite particle and interaction helpers should validate color lengths."""
    from types import SimpleNamespace

    particle = SimpleNamespace(
        shape=0,
        start_point=np.zeros(3),
        end_point=np.ones(3),
        start_dir=np.array([0.0, 0.0, 1.0]),
        ke=100.0,
    )
    interaction = SimpleNamespace(particles=[particle])

    with pytest.raises(ValueError, match="one per particle"):
        scatter_lite_particles([particle], color=[1.0, 2.0])
    with pytest.raises(ValueError, match="one per interaction"):
        scatter_lite_interactions([interaction], color=[1.0, 2.0])
