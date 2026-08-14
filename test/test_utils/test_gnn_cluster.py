"""Regression tests for GNN cluster utilities."""

import numpy as np
import pytest

from spine.data import ClusterLabelBatch, TensorBatch, TensorSchema
from spine.utils.gnn.cluster import (
    cluster_dedx,
    cluster_direction,
    form_clusters_batch,
    get_cluster_closest_label_batch,
    get_cluster_directions,
    get_cluster_features_base,
    get_cluster_label_batch,
    get_cluster_points_label,
    get_cluster_points_label_batch,
)


def test_cluster_batch_utilities_use_structured_particle_fields():
    """Resolve cluster associations without relying on historical columns."""
    data = TensorBatch(
        np.array(
            [
                [0, 0, 0, 0, 1, 4, 0],
                [0, 1, 0, 0, 1, 4, 0],
                [0, 2, 0, 0, 1, 8, 1],
            ],
            dtype=np.float32,
        ),
        counts=[3],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    labels = ClusterLabelBatch(
        data,
        {
            "particle": TensorBatch(np.array([41, 77]), counts=[2]),
            "group": TensorBatch(np.array([11, 12]), counts=[2]),
            "shape": TensorBatch(np.array([0, 1]), counts=[2]),
        },
    )
    coord_label = TensorBatch(
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 2, 0, 0, 2, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        counts=[2],
        has_batch_col=True,
        coord_cols=np.arange(1, 7),
        schema=TensorSchema(
            coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
            feature_fields={"time": (0,), "shape": (1,)},
        ),
    )

    clusters = form_clusters_batch(labels, column="cluster")
    groups = get_cluster_label_batch(labels, clusters, column="group")
    points = get_cluster_points_label_batch(
        labels, coord_label, clusters, random_order=False
    )

    assert [index.tolist() for index in clusters.index_list] == [[0, 1], [2]]
    assert groups.numpy_tensor().tolist() == [11, 12]
    np.testing.assert_allclose(
        points.numpy_tensor(),
        [[0, 0, 0, 1, 0, 0], [2, 0, 0, 2, 0, 0]],
    )


def test_closest_cluster_label_resolves_physical_group_id():
    """Map a physical group ID to its event-local coordinate-label row."""
    data = TensorBatch(
        np.array(
            [
                [0, 0, 0, 0, 1, 4, 0],
                [0, 2, 0, 0, 1, 8, 1],
            ],
            dtype=np.float32,
        ),
        counts=[2],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    truth = ClusterLabelBatch(
        data,
        {
            "particle": TensorBatch(np.array([41, 77]), counts=[2]),
            "group": TensorBatch(np.array([41, 41]), counts=[2]),
        },
    )
    coord_label = TensorBatch(
        np.array(
            [
                [0, 2, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
        counts=[2],
        has_batch_col=True,
        coord_cols=np.arange(1, 7),
        schema=TensorSchema(
            coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
            feature_fields={"time": (0,), "shape": (1,)},
        ),
    )
    clusters = form_clusters_batch(truth, column="cluster")
    labels = TensorBatch(np.array([2, 3]), counts=[2])

    adapted = get_cluster_closest_label_batch(
        truth,
        coord_label,
        clusters,
        labels,
        default=np.array([0, 0, 8, 9]),
    )

    assert adapted.numpy_tensor().tolist() == [9, 3]


def test_cluster_dedx_accepts_mixed_coordinate_dtypes():
    """Mixed start/voxel dtypes should not fail inside the anchored cdist path."""
    voxels = np.array(
        [[0.0, 1.0, 2.0], [0.0, 1.0, 3.0]],
        dtype=np.float32,
    )
    values = np.array([1.0, 2.0], dtype=np.float32)
    start = np.array([0.0, 1.0, 2.5], dtype=np.float64)

    dedx = cluster_dedx(voxels, values, start, 5.0, True)

    assert dedx == np.float32(3.0)


def test_cluster_directions_preserve_reference_point_dtype():
    """Directions must match the start/end-point dtype, not the voxel dtype."""
    voxels = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    starts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    clusts = [np.array([0, 1], dtype=np.int64)]

    directions = get_cluster_directions(voxels, starts, clusts)

    assert directions.dtype == starts.dtype
    np.testing.assert_allclose(directions, [[1.0, 0.0, 0.0]])


def test_cluster_direction_handles_duplicate_coordinates_during_optimization():
    """Coincident leading points must not divide by zero in PCA optimization."""
    voxels = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    direction = cluster_direction(
        voxels,
        np.zeros(3, dtype=np.float64),
        optimize=True,
    )

    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0])


def test_cluster_features_base_accepts_indexed_float32_coordinates():
    """Indexed cluster coordinate views should compile through Numba helpers."""
    data = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [2.0, 0.0, 0.0, 4.0],
            [2.0, 1.0, 0.0, 5.0],
            [3.0, 0.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    clusts = [
        np.array([0, 2, 1], dtype=np.int64),
        np.array([3, 5, 4], dtype=np.int64),
    ]

    feats = get_cluster_features_base(data[:, 1:], clusts)

    assert feats.shape == (2, 16)
    np.testing.assert_allclose(feats[0, :3], [1.0 / 3.0, 0.0, 2.0])
    np.testing.assert_allclose(feats[1, :3], [1.0 / 3.0, 0.0, 5.0])


def test_cluster_points_label_rejects_invalid_particle_id():
    coords = np.zeros((1, 3), dtype=np.float32)
    particle_ids = np.ones(1, dtype=np.float32)
    starts = np.zeros((1, 3), dtype=np.float32)
    ends = np.zeros((1, 3), dtype=np.float32)
    times = np.zeros(1, dtype=np.float32)

    with pytest.raises(IndexError, match="Invalid label index"):
        get_cluster_points_label(
            coords,
            particle_ids,
            starts,
            ends,
            times,
            [np.array([0], dtype=np.int64)],
            random_order=False,
        )
