"""Behavioral tests for the top-level GraphSPICE model and loss."""

from typing import Any

import pytest
import torch

from spine.constants import SHAPE_COL
from spine.data import TensorBatch
from spine.model.graph_spice import EdgeLoss, GraphSPICE, GraphSPICELoss


def test_graph_spice_rejects_misaligned_segmentation_labels():
    """Filtering must validate labels before applying their indexes to data."""
    model = object.__new__(GraphSPICE)
    model.shapes = [0]
    data = TensorBatch(torch.zeros((2, 5)), counts=[2], has_batch_col=True)
    seg_label = TensorBatch(torch.zeros((3, SHAPE_COL + 1)), counts=[3])

    with pytest.raises(ValueError, match="matching row counts"):
        model.filter_class(data, seg_label)


def test_graph_spice_loss_builds_explicit_loss_configuration():
    """The required loss block must resolve its named implementation."""
    loss = GraphSPICELoss(
        {"constructor": {}},
        {"name": "edge", "metric": None},
    )

    assert isinstance(loss.loss_fn, EdgeLoss)
    assert loss.constructor is None


def test_graph_spice_loss_requires_loss_configuration():
    """A Graph-SPICE model block alone cannot define its training loss."""
    # Deliberately bypass static argument checking to exercise the runtime
    # failure produced by the model manager when the configuration omits this
    # required block.
    loss_cls: Any = GraphSPICELoss
    with pytest.raises(TypeError, match="graph_spice_loss"):
        loss_cls({"constructor": {}})
