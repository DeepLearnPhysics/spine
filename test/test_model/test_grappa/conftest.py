"""Shared fixtures for graph-neural-network layer tests."""

import numpy as np
import pytest

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


@pytest.fixture
def graph_data():
    """Return a two-entry voxel table with three clusters."""
    rows = np.zeros((6, 5), dtype=np.float32)
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
    return TensorBatch(
        rows,
        counts=np.array([4, 2], dtype=np.int64),
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )


@pytest.fixture
def graph_labels(graph_data):
    """Return structured truth labels aligned with ``graph_data``."""
    rows = graph_data.numpy_tensor()
    compact = np.column_stack(
        (
            rows[:, :5],
            np.array([0, 0, 1, 1, 0, 0], dtype=np.float32),
            np.array([0, 0, 1, 1, 0, 0], dtype=np.float32),
        )
    )
    data = TensorBatch(
        compact,
        graph_data.counts,
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    particles = {
        "shape": TensorBatch(
            np.array([0, 1, 1], dtype=np.int64),
            counts=np.array([2, 1], dtype=np.int64),
        )
    }
    return ClusterLabelBatch(data, particles)


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
