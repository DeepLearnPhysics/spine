"""Regression tests for GNN cluster utilities."""

import numpy as np

from spine.constants import GROUP_COL, PART_COL
from spine.utils.gnn.cluster import (
    cluster_dedx,
    get_cluster_features_base,
    get_cluster_points_label,
)


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

    feats = get_cluster_features_base(data, clusts)

    assert feats.shape == (2, 16)
    np.testing.assert_allclose(feats[0, :3], [1.0 / 3.0, 0.0, 2.0])
    np.testing.assert_allclose(feats[1, :3], [1.0 / 3.0, 0.0, 5.0])


def test_cluster_points_label_can_use_group_identity():
    data = np.zeros((2, GROUP_COL + 1), dtype=np.float32)
    data[:, 1:4] = np.array(
        [
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    data[:, PART_COL] = [0, 1]
    data[:, GROUP_COL] = 1
    coord_label = np.array(
        [
            [0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )

    points = get_cluster_points_label(
        data,
        coord_label,
        [np.array([0, 1], dtype=np.int64)],
        random_order=False,
        use_group=True,
    )

    np.testing.assert_allclose(points[0, :3], [20.0, 0.0, 0.0])
    np.testing.assert_allclose(points[0, 3:], [20.0, 0.0, 0.0])
