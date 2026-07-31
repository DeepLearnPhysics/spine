"""Behavioral tests for clustering losses."""

import pytest
import torch

from spine.data import TensorBatch
from spine.model.graph_spice import EdgeLoss


def test_edge_loss_samples_balanced_classes_on_cpu():
    """Equal sampling must work without assuming a CUDA device."""
    loss_fn = EdgeLoss(
        loss="bce_logits",
        invert=False,
        equal_sampling=True,
        min_sample_edges=4,
        metric=None,
    )
    edge_logits = TensorBatch(
        torch.tensor([-2.0, -1.0, -0.5, -0.25, 1.0, 2.0]),
        counts=[6],
    )
    edge_labels = TensorBatch(
        torch.tensor([0, 0, 0, 0, 1, 1]),
        counts=[6],
    )
    cluster_labels = TensorBatch(torch.empty((0, 1)), counts=[0])

    result = loss_fn(cluster_labels, edge_logits, edge_labels)

    assert result["count"] == 8
    assert result["accuracy"] == 1.0
    assert torch.isfinite(result["loss"])


def test_edge_loss_returns_finite_zero_for_empty_graph():
    """An entry without graph edges must not produce a NaN loss."""
    loss_fn = EdgeLoss(invert=False)
    edge_logits = TensorBatch(
        torch.empty(0, requires_grad=True),
        counts=[0],
    )
    edge_labels = TensorBatch(torch.empty(0, dtype=torch.long), counts=[0])
    cluster_labels = TensorBatch(torch.empty((0, 1)), counts=[0])

    result = loss_fn(cluster_labels, edge_logits, edge_labels)

    assert result["count"] == 0
    assert result["accuracy"] == 1.0
    assert result["iou"] == 0.0
    torch.testing.assert_close(result["loss"], torch.tensor(0.0))


def test_edge_loss_rejects_mismatched_inputs():
    """Edge logits and labels must obey the shared edge contract."""
    loss_fn = EdgeLoss(metric=None)
    cluster_labels = TensorBatch(torch.empty((0, 1)), counts=[0])
    edge_logits = TensorBatch(torch.zeros(2), counts=[2])
    edge_labels = TensorBatch(torch.zeros(3, dtype=torch.long), counts=[3])

    with pytest.raises(ValueError, match="same length"):
        loss_fn(cluster_labels, edge_logits, edge_labels)


def test_edge_loss_rejects_nonbinary_labels():
    """Fractional or multiclass targets must not be silently truncated."""
    loss_fn = EdgeLoss(metric=None)
    cluster_labels = TensorBatch(torch.empty((0, 1)), counts=[0])
    edge_logits = TensorBatch(torch.zeros(2), counts=[2])
    edge_labels = TensorBatch(torch.tensor([0.0, 1.5]), counts=[2])

    with pytest.raises(ValueError, match="must be binary"):
        loss_fn(cluster_labels, edge_logits, edge_labels)
