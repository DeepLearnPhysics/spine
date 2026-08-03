"""Shared fixtures for construct tests."""

import numpy as np
import pytest

from spine.data import ClusterLabelData, TensorData
from spine.data.larcv.meta import ImageMeta3D


@pytest.fixture
def meta_cm():
    """Simple 3D image metadata with 2 cm voxels."""
    return ImageMeta3D(
        lower=np.zeros(3, dtype=np.float32),
        upper=np.full(3, 20.0, dtype=np.float32),
        size=np.full(3, 2.0, dtype=np.float32),
        count=np.full(3, 10, dtype=np.int64),
    )


@pytest.fixture
def points():
    """Small point cloud used by reconstructed builders."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def depositions():
    """Charge depositions aligned with the shared point cloud."""
    return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


def make_sparse_tensor(points: np.ndarray, values: np.ndarray) -> TensorData:
    """Build a self-describing sparse tensor product."""
    return TensorData(coords=points, features=values[:, None])


def make_label_tensor(
    points: np.ndarray,
    values: np.ndarray,
    clust_ids: list[int],
    part_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    inter_ids: list[int] | None = None,
    pids: list[int] | None = None,
    shapes: list[int] | None = None,
) -> ClusterLabelData:
    """Build structured cluster labels with one table row per voxel."""
    particle_ids = np.asarray(clust_ids if part_ids is None else part_ids)
    associations = np.arange(len(points), dtype=np.float32)
    associations[particle_ids < 0] = -1
    data = np.column_stack(
        (
            points,
            values,
            np.asarray(clust_ids),
            associations,
        )
    ).astype(np.float32)
    particles = {
        "particle": particle_ids,
        "group": np.asarray(clust_ids if group_ids is None else group_ids),
        "interaction": np.asarray(clust_ids if inter_ids is None else inter_ids),
        "pid": np.asarray([2] * len(points) if pids is None else pids),
        "shape": np.asarray([1] * len(points) if shapes is None else shapes),
    }
    return ClusterLabelData(data, particles)
