"""Behavioral tests for the top-level GraphSPICE model and loss."""

from copy import deepcopy
from typing import Any, cast

import pytest
import torch

from spine.constants import CLUST_COL, SHAPE_COL, SHOWR_SHP
from spine.data import IndexBatch, TensorBatch
from spine.model.graph_spice import EdgeLoss, GraphSPICE, GraphSPICELoss
from spine.utils.cluster.graph import ClusterGraphConstructor


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


def test_loss_edge_labels_match_legacy_network_labels():
    """Loss-side labeling must preserve the legacy network-side targets."""
    constructor_cfg = {
        "graph": {"name": "radius", "r": 1.9},
        "shapes": ["shower"],
        "edge_threshold": 0.1,
        "label_edges": True,
    }
    constructor = ClusterGraphConstructor(**deepcopy(constructor_cfg))
    constructor.graph_fn = lambda coords: torch.tensor(
        [[0], [1]],
        dtype=torch.long,
        device=coords.device,
    )
    constructor.kernel_fn = lambda left, right: torch.zeros(
        len(left),
        dtype=left.dtype,
        device=left.device,
    )

    coords = TensorBatch(torch.zeros((2, 3)), counts=[2])
    features = TensorBatch(torch.zeros((2, 2)), counts=[2])
    seg_tensor = torch.full((2, 1), SHOWR_SHP)
    seg_label = TensorBatch(seg_tensor, counts=[2])
    clust_tensor = torch.zeros((2, CLUST_COL + 1))
    clust_tensor[:, CLUST_COL] = 7
    clust_label = TensorBatch(clust_tensor, counts=[2])

    legacy_graph = constructor(coords, features, seg_label, clust_label)
    label_free_graph = constructor(coords, features, seg_label)
    loss = GraphSPICELoss(
        {"constructor": constructor_cfg},
        {"name": "edge", "metric": None},
    )
    edge_index = cast(TensorBatch, label_free_graph["edge_index"])
    node_clusts = cast(IndexBatch, label_free_graph["node_clusts"])
    edge_clusts = cast(IndexBatch, label_free_graph["edge_clusts"])
    derived_labels = loss.get_edge_labels(
        clust_label,
        edge_index,
        node_clusts,
        edge_clusts,
    )

    assert "edge_label" not in label_free_graph
    legacy_labels = cast(TensorBatch, legacy_graph["edge_label"])
    assert torch.equal(
        derived_labels.torch_tensor(),
        legacy_labels.torch_tensor(),
    )
