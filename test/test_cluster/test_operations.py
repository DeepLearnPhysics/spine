"""Regression tests for GNN cluster utilities."""

import numpy as np
import pytest

from spine.cluster import (
    break_clusters,
    cluster_dedx,
    cluster_dedx_dir,
    cluster_direction,
    form_clusters,
    form_clusters_batch,
    get_cluster_centers,
    get_cluster_closest_label_batch,
    get_cluster_closest_primary_label_batch,
    get_cluster_dedxs,
    get_cluster_dedxs_batch,
    get_cluster_directions,
    get_cluster_directions_batch,
    get_cluster_energies,
    get_cluster_features,
    get_cluster_features_base,
    get_cluster_features_batch,
    get_cluster_features_extended,
    get_cluster_label,
    get_cluster_label_batch,
    get_cluster_points_label,
    get_cluster_points_label_batch,
    get_cluster_primary_label_batch,
    get_cluster_sizes,
)
from spine.data import ClusterLabelBatch, TensorBatch, TensorSchema


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
            "multi": TensorBatch(np.array([[1, 2], [3, 4]]), counts=[2]),
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
    assert get_cluster_label_batch(labels, clusters, "multi").shape == (2, 2)
    assert get_cluster_primary_label_batch(labels, clusters, "shape").shape == (2,)
    starts = TensorBatch(points.numpy_tensor()[:, :3], counts=[2])
    assert get_cluster_directions_batch(labels, starts, clusters).shape == (2, 3)
    assert get_cluster_dedxs_batch(labels, starts, clusters).shape == (2,)
    assert get_cluster_features_batch(labels, clusters).shape == (2, 16)
    assert get_cluster_features_batch(labels, clusters, add_value=True).shape == (
        2,
        18,
    )
    assert get_cluster_features_batch(labels, clusters, add_shape=True).shape == (
        2,
        17,
    )
    with pytest.raises(TypeError, match="structured"):
        get_cluster_features_batch(data, clusters, add_shape=True)
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
    primary = get_cluster_closest_primary_label_batch(
        truth, coord_label, clusters, TensorBatch(np.ones(2), counts=[2])
    )
    assert primary.numpy_tensor().tolist() == [0.0, 1.0]

    # A malformed group association has no matching particle reference. The
    # closest-label helpers leave the supplied labels unchanged in that case.
    missing_group = ClusterLabelBatch(
        data,
        {
            "particle": TensorBatch(np.array([41, 77]), counts=[2]),
            "group": TensorBatch(np.array([99, 99]), counts=[2]),
        },
    )
    missing_clusters = form_clusters_batch(missing_group, column="cluster")
    unchanged = get_cluster_closest_label_batch(
        missing_group,
        coord_label,
        missing_clusters,
        labels,
        default=np.array([0, 0, 8, 9]),
    )
    np.testing.assert_array_equal(unchanged.numpy_tensor(), labels.numpy_tensor())
    unchanged_primary = get_cluster_closest_primary_label_batch(
        missing_group,
        coord_label,
        missing_clusters,
        TensorBatch(np.ones(2), counts=[2]),
    )
    np.testing.assert_array_equal(unchanged_primary.numpy_tensor(), np.ones(2))


def test_cluster_points_batch_rejects_missing_particle_coordinate():
    """Structured point extraction should reject clusters with no valid label row."""
    data = TensorBatch(
        np.array([[0, 0, 0, 0, 1, 4, -1]], dtype=np.float32),
        counts=[1],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    labels = ClusterLabelBatch(
        data,
        {
            "particle": TensorBatch(np.array([41]), counts=[1]),
            "group": TensorBatch(np.array([41]), counts=[1]),
        },
    )
    coordinates = TensorBatch(
        np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32),
        counts=[1],
        has_batch_col=True,
        coord_cols=np.arange(1, 7),
        schema=TensorSchema(
            coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
            feature_fields={"time": (0,), "shape": (1,)},
        ),
    )
    clusters = form_clusters_batch(labels, column="cluster")
    with pytest.raises(IndexError, match="no valid particle"):
        get_cluster_points_label_batch(labels, coordinates, clusters)


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

    assert get_cluster_points_label(
        coords, particle_ids, starts, ends, times, [], random_order=False
    ).shape == (0, 6)
    with pytest.raises(IndexError, match="Invalid label index"):
        get_cluster_points_label(
            coords,
            particle_ids,
            starts,
            ends,
            times,
            [np.empty(0, dtype=np.int64)],
            random_order=False,
        )


def test_cluster_points_label_selects_earliest_particle(monkeypatch):
    """Point labels should select the earliest constituent and optionally swap."""
    coords = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    particle_ids = np.array([0, 1], dtype=np.float32)
    starts = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    ends = np.array([[1, 0, 0], [3, 0, 0]], dtype=np.float32)
    times = np.array([2, 1], dtype=np.float32)
    clusts = [np.array([0, 1])]

    monkeypatch.setattr(np.random, "choice", lambda size: 1)
    points = get_cluster_points_label(
        coords, particle_ids, starts, ends, times, clusts, random_order=True
    )
    assert points.shape == (1, 6)


def test_cluster_array_algorithms_and_empty_contracts():
    """Core cluster reducers should cover NumPy, Torch, filtering, and empties."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [3, 1, 0]], dtype=np.float32)
    values = np.array([1, 2, 3, 4], dtype=np.float32)
    shapes = np.array([0, 0, 1, 1], dtype=np.int64)
    ids = np.array([0, 0, 1, 1], dtype=np.int64)
    clusts = [np.array([0, 1]), np.array([2, 3])]

    formed, sizes = form_clusters(ids[:, None], shapes=[1], shape_values=shapes)
    assert [c.tolist() for c in formed] == [[2, 3]]
    assert sizes.tolist() == [2]
    torch_formed, torch_sizes = form_clusters(
        __import__("torch").as_tensor(ids),
        shapes=[0],
        shape_values=__import__("torch").as_tensor(shapes),
    )
    assert torch_formed[0].tolist() == [0, 1]
    assert torch_sizes.tolist() == [2]
    with pytest.raises(ValueError, match="shape_values"):
        form_clusters(ids, shapes=[0])

    broken = break_clusters(coords, ids, clusts, 1.5, 1, 2.0)
    assert broken.shape == ids.shape
    np.testing.assert_array_equal(break_clusters(coords, ids, [], 1.5, 1, 2.0), ids)

    np.testing.assert_array_equal(get_cluster_label(shapes, clusts), [0, 1])
    np.testing.assert_allclose(
        get_cluster_centers(coords, clusts), [[0.5, 0, 0], [3, 0.5, 0]]
    )
    np.testing.assert_array_equal(get_cluster_sizes(coords, clusts), [2, 2])
    np.testing.assert_array_equal(get_cluster_energies(values, clusts), [3, 7])
    assert get_cluster_label(shapes, []).shape == (0,)
    assert get_cluster_centers(coords, []).shape == (0, 3)
    assert get_cluster_sizes(coords, []).shape == (0,)
    assert get_cluster_energies(values, []).shape == (0,)

    assert get_cluster_features(coords, clusts, values, shapes).shape == (2, 19)
    assert get_cluster_features_extended(None, shapes, clusts).shape == (2, 1)
    assert get_cluster_features_extended(values, None, clusts).shape == (2, 2)
    assert get_cluster_features_extended(values, shapes, []).shape == (0, 3)
    assert get_cluster_features_base(coords, []).shape == (0, 16)
    with pytest.raises(ValueError, match="Provide"):
        get_cluster_features_extended(None, None, clusts)


def test_cluster_direction_and_dedx_edge_cases():
    """Direction and local-deposition helpers should exercise optimization guards."""
    coords = np.array(
        [[0, 0, 0], [1, 0.1, 0], [2, -0.1, 0], [3, 0, 0]], dtype=np.float32
    )
    values = np.ones(4, dtype=np.float32)
    start = np.zeros(3, dtype=np.float32)
    clusts = [np.arange(4)]

    assert np.isfinite(cluster_direction(coords, start, 2.0, True)).all()
    np.testing.assert_array_equal(
        cluster_direction(np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.float32), start),
        np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        cluster_direction(coords[:1], start), np.array([1, 0, 0], dtype=np.float32)
    )
    assert np.isfinite(get_cluster_directions(coords, start[None], clusts)).all()
    assert get_cluster_directions(
        coords, np.empty((0, 3), dtype=np.float32), []
    ).shape == (0, 3)
    with pytest.raises(ValueError, match="three-dimensional"):
        get_cluster_directions(coords[:, :2], start[None, :2], clusts)
    with pytest.raises(ValueError, match="three-dimensional"):
        cluster_direction(coords[:, :2], start[:2])

    assert get_cluster_dedxs(coords, values, start[None], clusts, 2.0, True)[0] > 0
    assert get_cluster_dedxs(coords, values, np.empty((0, 3)), []).shape == (0,)
    assert cluster_dedx(coords, values, start, 0.1) == 0.0
    assert (
        cluster_dedx(np.zeros((2, 3), dtype=np.float32), values[:2], start, -1) == 0.0
    )
    with pytest.raises(ValueError, match="three-dimensional"):
        cluster_dedx(coords[:, :2], values, start[:2])

    result = cluster_dedx_dir(
        coords, values, start, np.array([1, 0, 0], dtype=np.float32)
    )
    assert len(result) == 5 and result[0] > 0
    assert (
        cluster_dedx_dir(
            coords,
            values,
            start + 0.1,
            np.array([1, 0, 0], dtype=np.float32),
            anchor=True,
        )[0]
        > 0
    )
    assert (
        cluster_dedx_dir(coords, values, start, np.array([-1, 0, 0], dtype=np.float32))[
            0
        ]
        == 0
    )
    assert (
        cluster_dedx_dir(
            coords, values, start, np.array([1, 0, 0], dtype=np.float32), 0.1
        )[0]
        == 0
    )
    with pytest.raises(ValueError, match="three-dimensional"):
        cluster_dedx_dir(coords[:, :2], values, start[:2], start[:2])
