"""Behavioral tests for GNN message-passing layers."""

import torch

from spine.data import TensorBatch
from spine.model.layer.gnn.model.layer.mlp import MLPNodeLayer
from spine.model.layer.gnn.model.meta import MetaLayerGNN

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
