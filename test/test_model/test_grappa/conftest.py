"""Shared fixtures for graph-neural-network layer tests."""

import numpy as np
import pytest

from spine.constants import PRINT_COL
from spine.data import IndexBatch, TensorBatch

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


@pytest.fixture
def graph_data():
    """Return a two-entry voxel table with three clusters."""
    rows = np.zeros((6, PRINT_COL + 1), dtype=np.float32)
    rows[:, 0] = (0, 0, 0, 0, 1, 1)
    rows[:, 1:4] = (
        (0, 0, 0),
        (0, 1, 0),
        (3, 0, 0),
        (3, 1, 0),
        (0, 0, 0),
        (0, 1, 0),
    )
    rows[:, 4] = 1.0
    rows[:, PRINT_COL] = (1, 1, 0, 0, 1, 1)
    return TensorBatch(rows, counts=np.array([4, 2], dtype=np.int64))


@pytest.fixture
def graph_clusters():
    """Return two clusters in the first entry and one in the second."""
    clusters = [
        np.array([0, 1], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
        np.array([4, 5], dtype=np.int64),
    ]
    return IndexBatch(
        clusters,
        spans=np.array([4, 2], dtype=np.int64),
        counts=np.array([2, 1], dtype=np.int64),
        single_counts=np.array([2, 2, 2], dtype=np.int64),
    )
