"""Module which contains a generic GNN message passing implementation."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch
from torch_geometric.nn import MetaLayer

from spine.data import TensorBatch
from spine.model.common.act_norm import norm_factory

from .factories import edge_layer_factory, global_layer_factory, node_layer_factory

__all__ = ["MetaLayerGNN"]

GNNOutput: TypeAlias = dict[str, TensorBatch]


class MetaLayerGNN(torch.nn.Module):
    """Completely generic message-passing GNN."""

    # Name of the model (as specified in the configuration)
    name = "meta"

    def __init__(
        self,
        node_feats: int = 0,
        node_layer: dict[str, Any] | None = None,
        node_pred: bool = True,
        edge_feats: int = 0,
        edge_layer: dict[str, Any] | None = None,
        edge_pred: bool = True,
        global_feats: int = 0,
        global_layer: dict[str, Any] | None = None,
        global_pred: bool = True,
        num_mp: int = 3,
        input_normalization: str | dict[str, Any] = "batch_norm",
    ) -> None:
        """Initializes the message passing network.

        Parameters
        ----------
        node_feats : int, default 0
            Number of node features
        node_layer : dict, optional
            Configuration of the node features update layer
        node_pred : bool, default True
            If `True`, return the node features (used for predictions)
        edge_feats : int, default 0
            Number of edge features
        edge_layer : dict, optional
            Configuration of the edge features update layer
        edge_pred : bool, default True
            If `True`, return the edge features (used for predictions)
        global_feats : int, default 0
            Number of global features
        global_layer : dict, optional
            Configuration of the global features update layer
        global_pred : bool, default True
            If `True`, return the global features (used for predictions)
        num_mp : int, default 3
            Number of message passing steps (node/edge/global feature updates)
        input_normalization : union[str, dict], default 'batch_norm'
            Input node/edge/global feature ormalization function configuration
        """
        # Initialize the parent class
        super().__init__()

        # Store the attributes
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.global_feats = global_feats
        self.node_pred = node_pred
        self.edge_pred = edge_pred
        self.global_pred = global_pred
        self.num_mp = num_mp
        if num_mp < 1:
            raise ValueError(f"`num_mp` must be positive, got {num_mp}.")

        # Check that at least one of the output features is needed
        if not node_pred and not edge_pred and not global_pred:
            raise ValueError(
                "Must request at least one type of GNN features to be output."
            )

        # Intialize the input normalization layers
        self.node_bn, self.edge_bn, self.global_bn = None, None, None
        if node_feats > 0:
            self.node_bn = norm_factory(input_normalization, node_feats)
        if edge_feats > 0:
            self.edge_bn = norm_factory(input_normalization, edge_feats)
        if global_feats > 0:
            self.global_bn = norm_factory(input_normalization, global_feats)

        # Loop over the number of message passing steps, initialize the
        # metalayer which updates the features at each step
        self.mp_layers = torch.nn.ModuleList()
        node_nf, edge_nf, glob_nf = (node_feats, edge_feats, global_feats)

        for layer_index in range(self.num_mp):
            # Initialize the edge update layer
            edge_model = None
            if edge_layer is not None:
                edge_model = edge_layer_factory(edge_layer, node_nf, edge_nf, glob_nf)
                edge_nf = edge_model.feature_size

            # Initialize the node update layer
            node_model = None
            if node_layer is not None:
                if (node_pred or global_pred) or layer_index < (self.num_mp - 1):
                    node_model = node_layer_factory(
                        node_layer, node_nf, edge_nf, glob_nf
                    )
                    node_nf = node_model.feature_size

            # Initialize the global update layer
            global_model = None
            if global_layer is not None:
                if global_pred or layer_index < (self.num_mp - 1):
                    global_model = global_layer_factory(global_layer, node_nf, glob_nf)
                    glob_nf = global_model.feature_size

            # Build the complete metalayer
            self.mp_layers.append(MetaLayer(edge_model, node_model, global_model))

        # Store the feature size of each of the outputs
        self.node_feature_size = node_nf
        self.edge_feature_size = edge_nf
        self.global_feature_size = glob_nf

    def forward(
        self,
        node_feats: TensorBatch,
        edge_index: torch.Tensor,
        edge_feats: TensorBatch | None,
        glob_feats: TensorBatch | None,
        batch: torch.Tensor,
    ) -> GNNOutput:
        """Run the message passing steps on one batch of data.

        Parameters
        ----------
        node_feats : TensorBatch
            (C) Batch of node features
        edge_index : torch.Tensor
            (2, E) Incidence matrix
        edge_feats : TensorBatch
            (E) Edge features
        glob_feats : TensorBatch
            (B) Global features
        batch : torch.Tensor
            (B) Batch ID of each node in the batched graph
        """
        # Pass input through the input normalization layer
        x = node_feats.torch_tensor()
        edge_tensor = x.new_empty((edge_index.shape[1], 0))
        u: torch.Tensor | None = None
        if self.node_bn is not None:
            x = self.node_bn(x)
        if edge_feats is not None:
            edge_tensor = edge_feats.torch_tensor()
            if self.edge_bn is not None:
                edge_tensor = self.edge_bn(edge_tensor)
        if glob_feats is not None:
            u = glob_feats.torch_tensor()
            if self.global_bn is not None:
                u = self.global_bn(u)

        # Loop over the message passing steps, update the graph features
        for layer in self.mp_layers:
            x, edge_tensor, u = layer(
                x,
                edge_index,
                edge_tensor,
                u,
                batch,
            )

        # Initialize and return result dictionary
        result: GNNOutput = {}
        if self.mp_layers[0].node_model is not None and self.node_pred:
            result["node_features"] = TensorBatch(x, node_feats.counts)
        if self.mp_layers[0].edge_model is not None and self.edge_pred:
            if edge_tensor is None:
                raise RuntimeError("Edge update did not produce edge features.")
            if edge_feats is None:
                edge_batches = batch[edge_index[0]]
                edge_counts = torch.bincount(
                    edge_batches,
                    minlength=node_feats.batch_size,
                )
            else:
                edge_counts = edge_feats.counts
            result["edge_features"] = TensorBatch(
                edge_tensor,
                edge_counts,
            )
        if self.mp_layers[0].global_model is not None and self.global_pred:
            if u is None:
                raise RuntimeError("Global update did not produce global features.")
            global_counts = (
                [1] * node_feats.batch_size if glob_feats is None else glob_feats.counts
            )
            result["global_features"] = TensorBatch(u, global_counts)

        return result
