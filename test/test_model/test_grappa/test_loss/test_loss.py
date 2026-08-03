"""Focused tests for GNN loss-label construction."""

import numpy as np
import torch

from spine.data import (
    ClusterLabelBatch,
    EdgeIndexBatch,
    IndexBatch,
    Meta,
    TensorBatch,
)
from spine.model.grappa.loss import NodeVertexLoss
from spine.utils.gnn.evaluation import edge_assignment_forest_batch


def test_forest_assignment_builds_valid_spanning_tree_labels():
    edge_index = EdgeIndexBatch(
        np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int64),
        counts=np.array([3], dtype=np.int64),
        spans=np.array([3], dtype=np.int64),
        directed=True,
    )
    edge_prediction = TensorBatch(
        np.array(
            [
                [0.0, 3.0],
                [3.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        ),
        counts=np.array([3], dtype=np.int64),
    )
    group_ids = TensorBatch(
        np.zeros(3, dtype=np.int64),
        counts=np.array([3], dtype=np.int64),
    )

    assignments, valid_mask = edge_assignment_forest_batch(
        edge_index,
        edge_prediction,
        group_ids,
    )

    assert assignments.numpy_tensor().sum() == 2
    assert valid_mask.numpy_tensor().sum() == 2


def test_vertex_loss_normalizes_each_batch_entry_with_its_own_meta():
    """Normalize vertex labels with the image dimensions of their entry."""
    data = TensorBatch(
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        ),
        counts=[1, 1],
        coord_cols=np.arange(1, 4),
    )
    particles = {
        "interaction_primary": TensorBatch(np.ones(2, dtype=np.int64), counts=[1, 1]),
        "vertex": TensorBatch(
            np.array(
                [
                    [50.0, 25.0, 10.0],
                    [100.0, 50.0, 20.0],
                ],
                dtype=np.float32,
            ),
            counts=[1, 1],
        ),
    }
    clust_label = ClusterLabelBatch(data, particles)
    clusts = IndexBatch(
        [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ],
        spans=[1, 1],
        counts=[1, 1],
        single_counts=[1, 1],
    )
    node_pred = TensorBatch(
        torch.tensor(
            [
                [0.0, 1.0, 0.5, 0.5, 0.5],
                [0.0, 1.0, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        ),
        counts=[1, 1],
    )
    meta = [
        Meta(
            lower=np.zeros(3),
            upper=np.array([100.0, 50.0, 20.0]),
            size=np.ones(3),
            count=np.array([100, 50, 20]),
        ),
        Meta(
            lower=np.zeros(3),
            upper=np.array([200.0, 100.0, 40.0]),
            size=np.ones(3),
            count=np.array([200, 100, 40]),
        ),
    ]

    result = NodeVertexLoss(
        only_contained=False,
        normalize_positions=True,
        return_vertex_labels=True,
    )(clust_label, clusts, node_pred, meta=meta)

    torch.testing.assert_close(result["reg_loss"], torch.tensor(0.0))
    np.testing.assert_allclose(
        result["labels"].numpy_tensor(),
        np.full((2, 3), 0.5),
    )
