"""Behavioral tests for GNN message-passing layers."""

import pytest
import torch

from spine.data import TensorBatch
from spine.model.grappa.message_passing.factories import global_layer_factory
from spine.model.grappa.message_passing.layers import (
    AGNNConvNodeLayer,
    EConvNodeLayer,
    GATConvNodeLayer,
    NNConvNodeLayer,
)
from spine.model.grappa.message_passing.layers.mlp import (
    MLPEdgeLayer,
    MLPGlobalLayer,
    MLPNodeLayer,
)
from spine.model.grappa.message_passing.meta import MetaLayerGNN

MLP_CONFIG = {
    "depth": 1,
    "width": 4,
    "activation": "relu",
    "normalization": "none",
}


def test_attention_node_layer_uses_message_width():
    layer = MLPNodeLayer(
        node_in=3,
        edge_in=2,
        glob_in=1,
        message_mlp=MLP_CONFIG,
        aggr_mlp=MLP_CONFIG,
        attention=True,
    )
    nodes = torch.randn(3, 3)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_features = torch.randn(3, 2)
    global_features = torch.randn(1, 1)
    batch = torch.zeros(3, dtype=torch.long)

    output = layer(
        nodes,
        edges,
        edge_features,
        global_features,
        batch,
    )

    assert output.shape == (3, 4)


def test_meta_layer_runs_node_and_edge_updates():
    model = MetaLayerGNN(
        node_feats=3,
        edge_feats=2,
        global_feats=0,
        edge_layer={"name": "mlp", "mlp": MLP_CONFIG},
        node_layer={
            "name": "mlp",
            "message_mlp": MLP_CONFIG,
            "aggr_mlp": MLP_CONFIG,
        },
        node_pred=True,
        edge_pred=True,
        global_pred=False,
        num_mp=2,
        input_normalization="none",
    )
    node_features = TensorBatch(torch.randn(3, 3), counts=[3])
    edge_features = TensorBatch(torch.randn(3, 2), counts=[3])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    batch = torch.zeros(3, dtype=torch.long)

    output = model(
        node_features,
        edge_index,
        edge_features,
        None,
        batch,
    )

    assert output["node_features"].shape == (3, 4)
    assert output["edge_features"].shape == (3, 4)


def test_torch_geometric_node_layers_share_the_node_update_contract():
    """All maintained PyG adapters accept the canonical graph tensors."""
    nodes = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_features = torch.randn(3, 2)
    layers = [
        (
            AGNNConvNodeLayer(
                node_in=3,
                edge_in=2,
                glob_in=0,
                normalization="none",
            ),
            3,
        ),
        (
            GATConvNodeLayer(
                node_in=3,
                edge_in=2,
                glob_in=0,
                out_channels=4,
                normalization="none",
            ),
            4,
        ),
        (
            EConvNodeLayer(
                node_in=3,
                edge_in=2,
                glob_in=0,
                mlp=MLP_CONFIG,
            ),
            4,
        ),
    ]

    for layer, width in layers:
        output = layer(nodes, edge_index, edge_features)
        assert output.shape == (3, width)

    nnconv = NNConvNodeLayer(
        node_in=3,
        edge_in=2,
        glob_in=0,
        out_channels=4,
        mlp=MLP_CONFIG,
    )
    assert nnconv(nodes, edge_index, edge_features).shape == (3, 4)


def test_mlp_edge_and_global_layers_use_graph_features():
    """MLP updates concatenate graph context and aggregate per batch."""
    nodes = torch.randn(3, 3)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_features = torch.randn(3, 2)
    globals_ = torch.randn(1, 1)
    batch = torch.zeros(3, dtype=torch.long)

    edge_layer = MLPEdgeLayer(3, 2, 1, MLP_CONFIG)
    edge_output = edge_layer(
        nodes[edges[0]],
        nodes[edges[1]],
        edge_features,
        globals_,
        batch[edges[0]],
    )
    assert edge_output.shape == (3, 4)

    global_layer = MLPGlobalLayer(3, 1, MLP_CONFIG)
    global_output = global_layer(nodes, edges, edge_features, globals_, batch)
    assert global_output.shape == (1, 4)
    assert isinstance(
        global_layer_factory(
            {"name": "mlp", "mlp": MLP_CONFIG},
            node_in=3,
            glob_in=1,
        ),
        MLPGlobalLayer,
    )


def test_meta_layer_validates_iterations_and_requested_outputs():
    """MetaLayerGNN requires work to perform and a public output contract."""
    with pytest.raises(ValueError, match="num_mp"):
        MetaLayerGNN(3, 2, num_mp=0)
    with pytest.raises(ValueError, match="at least one"):
        MetaLayerGNN(
            3,
            2,
            node_pred=False,
            edge_pred=False,
            global_pred=False,
        )


def test_meta_layer_updates_and_returns_global_features():
    """Global features participate in every update and retain batch counts."""
    model = MetaLayerGNN(
        node_feats=3,
        edge_feats=2,
        global_feats=1,
        edge_layer={"name": "mlp", "mlp": MLP_CONFIG},
        node_layer={
            "name": "mlp",
            "message_mlp": MLP_CONFIG,
            "aggr_mlp": MLP_CONFIG,
        },
        global_layer={"name": "mlp", "mlp": MLP_CONFIG},
        node_pred=True,
        edge_pred=True,
        global_pred=True,
        num_mp=1,
        input_normalization="none",
    )
    nodes = TensorBatch(torch.randn(3, 3), counts=[2, 1])
    edges = torch.tensor([[0, 2], [1, 2]])
    edge_features = TensorBatch(torch.randn(2, 2), counts=[1, 1])
    globals_ = TensorBatch(torch.randn(2, 1), counts=[1, 1])
    batch = torch.tensor([0, 0, 1])

    output = model(nodes, edges, edge_features, globals_, batch)

    assert output["node_features"].shape == (3, 4)
    assert output["edge_features"].shape == (2, 4)
    assert output["global_features"].shape == (2, 4)
    assert output["global_features"].counts.tolist() == [1, 1]


def test_meta_edge_output_derives_counts_without_input_features():
    """Generated edge features derive event counts from source node batches."""
    model = MetaLayerGNN(
        node_feats=3,
        edge_feats=0,
        global_feats=0,
        edge_layer={"name": "mlp", "mlp": MLP_CONFIG},
        node_layer=None,
        global_layer=None,
        node_pred=False,
        edge_pred=True,
        global_pred=False,
        num_mp=1,
        input_normalization="none",
    )
    nodes = TensorBatch(torch.randn(3, 3), counts=[2, 1])
    edges = torch.tensor([[0, 2], [1, 2]])
    batch = torch.tensor([0, 0, 1])

    output = model(nodes, edges, None, None, batch)

    assert output["edge_features"].counts.tolist() == [1, 1]
