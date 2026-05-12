"""Regression tests for GNN cluster utilities."""

import numpy as np

from spine.utils.gnn.cluster import cluster_dedx


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
