"""Focused tests for UResNet model and loss contracts."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spine.constants import GHOST_SHP
from spine.data import TensorBatch
from spine.model.uresnet import SegmentationLoss


def make_loss(*, num_classes=2, ghost=False, balance_loss=False):
    """Build a segmentation loss without constructing a CNN."""
    loss = object.__new__(SegmentationLoss)
    torch.nn.Module.__init__(loss)
    loss.num_classes = num_classes
    loss.ghost = ghost
    loss.ghost_label = -1
    loss.alpha = 1.0
    loss.beta = 1.0
    loss.balance_loss = balance_loss
    loss.upweight_points = False
    loss.upweight_radius = 20.0
    loss.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    return loss


def make_labels(values):
    """Build one batch of sparse labels with the requested class values."""
    labels = torch.tensor(values, dtype=torch.float32)
    return TensorBatch(labels, counts=torch.tensor([len(values)]))


def make_logits(values):
    """Build one batch of differentiable logits."""
    logits = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    return TensorBatch(logits, counts=torch.tensor([len(values)]))


def test_ghost_loss_consumes_helper_result_and_tracks_filtered_weights():
    loss_fn = make_loss(ghost=True, balance_loss=True)
    labels = make_labels([0, 1, GHOST_SHP])
    segmentation = make_logits([[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    ghost = make_logits([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])

    result = loss_fn(labels, segmentation, ghost=ghost)

    assert torch.isfinite(result["loss"])
    assert result["weights"].counts.tolist() == [2]
    result["loss"].backward()


def test_ghost_loss_requires_ghost_logits():
    loss_fn = make_loss(ghost=True)
    labels = make_labels([0])
    segmentation = make_logits([[3.0, 0.0]])

    with pytest.raises(ValueError, match="ghost.*logits"):
        loss_fn(labels, segmentation)


def test_segmentation_loss_rejects_class_equal_to_num_classes():
    loss_fn = make_loss(num_classes=2)
    labels = make_labels([2])
    segmentation = make_logits([[3.0, 0.0]])

    with pytest.raises(ValueError, match=r"between 0 and 1"):
        loss_fn(labels, segmentation)


def test_segmentation_loss_rejects_numpy_backed_inputs():
    loss_fn = make_loss()
    labels = TensorBatch(
        np.zeros(1, dtype=np.float32),
        counts=np.array([1]),
    )
    segmentation = make_logits([[3.0, 0.0]])

    with pytest.raises(TypeError, match="not backed by a torch.Tensor"):
        loss_fn(labels, segmentation)


def test_point_weights_are_tensors_and_do_not_mutate_input_weights():
    loss_fn = make_loss()
    loss_fn.upweight_points = True
    loss_fn.get_distance_weights = lambda *_: torch.tensor([3.0, 4.0])

    labels = make_labels([0, 1])
    segmentation = make_logits([[3.0, 0.0], [0.0, 3.0]])
    input_weights = TensorBatch(
        torch.tensor([2.0, 2.0]),
        counts=torch.tensor([2]),
    )

    result = loss_fn(
        labels,
        segmentation,
        point_label=labels,
        weights=input_weights,
    )

    assert torch.equal(input_weights.tensor, torch.tensor([2.0, 2.0]))
    assert torch.equal(result["weights"].tensor, torch.tensor([6.0, 8.0]))


def test_empty_segmentation_loss_remains_differentiable():
    loss_fn = make_loss()
    logits = torch.empty((0, 2), requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)

    loss, accuracy, _, _ = loss_fn.get_loss_accuracy(logits, labels)

    assert isinstance(loss, torch.Tensor)
    assert accuracy == 1.0
    loss.backward()
