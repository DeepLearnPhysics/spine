"""Direct tests for generic GrapPA graph-evaluation helpers."""

import numpy as np
import pytest
import torch

from spine.data import EdgeIndexBatch, TensorBatch
from spine.model.grappa.evaluation import (
    adjacency_matrix,
    cluster_to_voxel_label,
    clustering_metrics,
    edge_assignment_forest,
    edge_assignment_forest_batch,
    edge_assignment_score,
    edge_purity_mask,
    grouping_loss,
    node_assignment_bipartite,
    node_assignment_score,
    node_purity_mask,
    node_purity_mask_batch,
    primary_assignment,
    primary_assignment_batch,
    voxel_efficiency_bipartite,
)


def test_cached_torch_targets_cross_numpy_evaluation_boundary():
    """Torch-backed cached IDs should reach NumPy helpers as integers."""
    counts = torch.tensor([3], dtype=torch.long)
    groups = TensorBatch(torch.tensor([0.0, 0.0, 1.0]), counts=counts)
    primaries = TensorBatch(torch.tensor([1.0, 0.0, 0.0]), counts=counts)

    node_valid = node_purity_mask_batch(groups, primaries)
    np.testing.assert_array_equal(node_valid, [True, True, False])

    edge_index = EdgeIndexBatch(
        torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        counts=torch.tensor([3], dtype=torch.long),
        spans=torch.tensor([3], dtype=torch.long),
        directed=True,
    )
    edge_prediction = TensorBatch(
        torch.tensor([[0.0, 3.0], [3.0, 0.0], [0.0, 3.0]]),
        counts=torch.tensor([3], dtype=torch.long),
    )
    forest_groups = TensorBatch(torch.zeros(3), counts=counts)
    edge_target, edge_valid = edge_assignment_forest_batch(
        edge_index,
        edge_prediction,
        forest_groups,
    )

    assert edge_target.is_numpy
    assert edge_valid.is_numpy
    assert edge_target.numpy_tensor().sum() == 2
    assert edge_valid.numpy_tensor().sum() == 2


def test_primary_bipartite_and_batch_assignments():
    """Primary and bipartite assignments should cover grouped and direct modes."""
    logits = np.array([[0.0, 2.0], [2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_equal(primary_assignment(logits), [True, False, True])
    np.testing.assert_array_equal(
        primary_assignment(logits, np.array([0, 0, 1])), [True, False, True]
    )
    batched = primary_assignment_batch(TensorBatch(logits, counts=[3]))
    np.testing.assert_array_equal(batched.tensor, [True, False, True])

    edges = np.array([[0, 1], [2, 1]], dtype=np.int64)
    labels = np.array([0.2, 0.9], dtype=np.float32)
    np.testing.assert_array_equal(
        node_assignment_bipartite(edges, labels, np.array([0, 2]), 4),
        [0, 2, 2, 3],
    )


def test_forest_grouping_losses_and_adjacency():
    """Forest and grouping helpers should cover empty groups and all loss modes."""
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64)
    logits = np.array([[0, 3], [0, 3], [3, 0], [3, 0]], dtype=np.float32)
    groups = np.array([0, 0, 1], dtype=np.int64)
    assignment, valid = edge_assignment_forest(edges, logits, groups)
    assert assignment.shape == valid.shape == (4,)
    empty = edge_assignment_forest(
        np.empty((0, 2), dtype=np.int64), np.empty((0, 2)), np.arange(2)
    )
    assert empty[0].shape == empty[1].shape == (0,)
    # Every node in a separate group exercises groups without internal edges.
    edge_assignment_forest(edges, logits, np.arange(3))

    pred = np.array([0.8, 0.2], dtype=np.float32)
    target = np.array([1, 0], dtype=bool)
    for mode in ("ce", "l1", "l2"):
        assert np.isfinite(grouping_loss(pred, target, mode))
    with pytest.raises(ValueError, match="not recognized"):
        grouping_loss(pred, target, "bad")
    adjacency = adjacency_matrix(edges, 3)
    assert adjacency.shape == (3, 3)
    assert adjacency.diagonal().all()


def test_score_assignment_purity_and_metrics():
    """Score grouping, purity masks, and voxel metrics should cover edge cases."""
    edges = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]], dtype=np.int64)
    logits = np.array(
        [[0, 4], [0, 4], [0, 3], [0, 3], [4, 0], [4, 0]], dtype=np.float32
    )
    selected, groups, score = edge_assignment_score(edges, logits, 3)
    assert selected.shape[1] == 2
    assert groups.shape == (3,)
    assert np.isfinite(score)
    np.testing.assert_array_equal(node_assignment_score(edges, logits, 3), groups)

    # Track restrictions cover both secondary endpoint orientations and reuse.
    edge_assignment_score(edges, logits, 3, track_node=np.array([True, False, True]))
    reuse_edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64)
    reuse_logits = np.tile(np.array([[0.0, 5.0]], dtype=np.float32), (4, 1))
    edge_assignment_score(
        reuse_edges, reuse_logits, 3, track_node=np.array([True, False, True])
    )
    empty = edge_assignment_score(np.empty((0, 2), dtype=np.int64), np.empty((0, 2)), 2)
    assert empty[0].shape == (0, 2)

    pure = node_purity_mask(np.array([0, 0, 1]), np.array([1, 0, 0]))
    np.testing.assert_array_equal(pure, [True, True, False])
    edge_pure = edge_purity_mask(
        edges,
        np.array([0, 1, 2]),
        np.array([0, 0, 1]),
        np.array([1, 0, 0]),
    )
    assert edge_pure.shape == (6,)

    clusts = [np.array([0, 1]), np.array([2]), np.array([3, 4])]
    truth = np.array([0, 0, 1])
    prediction = np.array([0, 0, 1])
    assert len(clustering_metrics(clusts, truth, prediction)) == 5
    np.testing.assert_array_equal(
        cluster_to_voxel_label(clusts, prediction), [0, 0, 0, 1, 1]
    )
    assert voxel_efficiency_bipartite(clusts, truth, prediction, [0, 2]) == 1.0
    assert voxel_efficiency_bipartite(clusts, truth, prediction, [0, 1, 2]) == 1.0
