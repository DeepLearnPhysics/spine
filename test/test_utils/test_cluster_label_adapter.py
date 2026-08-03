"""Regression tests for structured cluster-label adaptation."""

import numpy as np

from spine.data import ClusterLabelBatch, TensorBatch, TensorData
from spine.utils.cluster.label import ClusterLabelAdapter


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
