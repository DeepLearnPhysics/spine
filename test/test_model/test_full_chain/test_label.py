"""Regression tests for structured cluster-label adaptation."""

import numpy as np
import pytest

from spine.data import ClusterLabelBatch, ClusterLabelData, TensorBatch, TensorData
from spine.model.full_chain.label import ClusterLabelAdapter


def test_adapter_preserves_invalid_cluster_ids_across_events():
    """A global event offset must never turn invalid ``-1`` IDs positive."""
    compact = TensorBatch(
        np.asarray(
            [
                [0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
        counts=[1, 1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    cluster_label = ClusterLabelBatch(compact)
    semantic_label = TensorBatch.from_data_list(
        [
            TensorData(np.asarray([0], dtype=np.float32), coords=np.zeros((1, 3))),
            TensorData(np.asarray([0], dtype=np.float32), coords=np.zeros((1, 3))),
        ]
    )
    semantic_prediction = TensorBatch(np.asarray([0, 5], dtype=np.int64), counts=[1, 1])

    adapted = ClusterLabelAdapter()(cluster_label, semantic_label, semantic_prediction)

    assert adapted.cluster_ids.data[0] >= 0
    assert adapted.cluster_ids.data[1] == -1
    assert adapted.data.schema == cluster_label.data.schema


def adapter_for_numpy():
    """Return an adapter initialized for direct NumPy event processing."""
    adapter = ClusterLabelAdapter()
    adapter.torch = False
    adapter.dtype = np.dtype(np.float32)
    adapter.device = None
    adapter._offset = 0
    return adapter


def event_products(coords, truth, cluster_coords=None, cluster_features=None):
    """Build event-level semantic and compact cluster label products."""
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    truth = np.asarray(truth, dtype=np.float32)
    semantic = TensorData(truth, coords=coords)
    if cluster_coords is None:
        cluster_coords = np.empty((0, 3), dtype=np.float32)
    if cluster_features is None:
        cluster_features = np.empty((0, 2), dtype=np.float32)
    clusters = ClusterLabelData(
        coords=np.asarray(cluster_coords, dtype=np.float32),
        features=np.asarray(cluster_features, dtype=np.float32),
    )
    return clusters, semantic


def test_adapter_empty_dummy_deghost_and_validation_paths():
    """Direct event adaptation should handle each empty and alignment contract."""
    adapter = adapter_for_numpy()
    clusters, semantic = event_products([], [])
    assert adapter._process(clusters, semantic, np.empty(0, dtype=int)).shape == (0, 5)

    clusters, semantic = event_products([[0, 0, 0]], [0])
    assert adapter._process(
        clusters, semantic, np.empty(0, dtype=int), np.empty(0, dtype=int)
    ).shape == (0, 5)
    dummy = adapter._process(clusters, semantic, np.array([0]))
    assert np.all(dummy.features == -1)

    clusters, semantic = event_products(
        [[0, 0, 0]], [0], [[0, 0, 0], [1, 0, 0]], [[1, 0], [1, 1]]
    )
    with pytest.raises(ValueError, match="exactly"):
        adapter._process(clusters, semantic, np.array([0]))


def test_adapter_numpy_deghost_expansion_and_no_compatible_cluster():
    """NumPy adaptation should expand predictions and retain deghosted coordinates."""
    adapter = adapter_for_numpy()
    clusters, semantic = event_products(
        [[0, 0, 0], [1, 0, 0]],
        [0, 5],
        [[0, 0, 0]],
        [[1, 0]],
    )
    adapted = adapter._process(
        clusters, semantic, np.array([0]), np.array([0], dtype=np.int64)
    )
    assert adapted.shape == (1, 5)

    # A track prediction has no compatible true shower coordinate to borrow.
    clusters, semantic = event_products([[0, 0, 0]], [0], [[0, 0, 0]], [[1, 0]])
    adapted = adapter._process(clusters, semantic, np.array([1]))
    assert adapted.features[0, 1] == -1


def test_adapter_numpy_dispatch_helpers():
    """NumPy dispatch methods should retain their documented return contracts."""
    adapter = adapter_for_numpy()
    values = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0]], dtype=np.float32)
    minimum, indexes = adapter._min(values, 1)
    np.testing.assert_array_equal(minimum, [0.0, 0.0])
    np.testing.assert_array_equal(indexes, [2, 2])
    assert adapter._compute_distances(values, values).shape == (2, 2)
