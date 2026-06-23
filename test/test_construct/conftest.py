"""Shared fixtures for construct tests."""

import numpy as np
import pytest

from spine.constants import (
    CLUST_COL,
    COORD_COLS,
    GROUP_COL,
    INTER_COL,
    PART_COL,
    PID_COL,
    SHAPE_COL,
    VALUE_COL,
)
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


def make_sparse_tensor(points: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Build a sparse tensor with coordinate and value columns."""
    tensor = np.zeros((len(points), VALUE_COL + 1), dtype=np.float32)
    tensor[:, COORD_COLS] = points
    tensor[:, VALUE_COL] = values
    return tensor


def make_label_tensor(
    points: np.ndarray,
    values: np.ndarray,
    clust_ids: list[int],
    part_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    inter_ids: list[int] | None = None,
    pids: list[int] | None = None,
    shapes: list[int] | None = None,
) -> np.ndarray:
    """Build a cluster-label tensor with the columns used by construct."""
    tensor = np.full((len(points), 17), -1.0, dtype=np.float32)
    tensor[:, COORD_COLS] = points
    tensor[:, VALUE_COL] = values
    tensor[:, CLUST_COL] = clust_ids
    tensor[:, PART_COL] = clust_ids if part_ids is None else part_ids
    tensor[:, GROUP_COL] = clust_ids if group_ids is None else group_ids
    tensor[:, INTER_COL] = clust_ids if inter_ids is None else inter_ids
    tensor[:, PID_COL] = 2 if pids is None else pids
    tensor[:, SHAPE_COL] = 1 if shapes is None else shapes
    return tensor
