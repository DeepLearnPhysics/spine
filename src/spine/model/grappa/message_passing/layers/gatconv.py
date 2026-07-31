"""Module which defines a graph node feature update based on GATConv."""

from typing import Any

import torch
from torch_geometric.nn import GATConv

from spine.model.common.act_norm import act_factory, norm_factory

__all__ = ["GATConvNodeLayer"]


class GATConvNodeLayer(torch.nn.Module):
    """GATConv module for extracting graph node features.

    This model simply takes a simple attention-based convolution of a node
    with all of its neighbors to update the initial node feature vector (N_c),
    returning an updated (N_o) vector.

    Source: https://arxiv.org/abs/1710.10903
    """

    # Name of the node layer (as specified in the configuration)
    name = "gatconv"

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        glob_in: int,
        out_channels: int,
        activation: str | dict[str, Any] = "relu",
        normalization: str | dict[str, Any] = "batch_norm",
        **kwargs: Any,
    ) -> None:
        """Initialize the MLPs which are used to update the node features.

        Parameters
        ----------
        node_in : int
            Number of input node features
        edge_in : int
            Number of input edge features
        glob_in : int
            Number of input global features for the graph
        out_channels : int
            Number of output node features
        activation : Union[str, dict], default 'relu'
            Activation function configuration
        normalization : Union[str, dict], default 'batch_norm'
            Normalization function configuration
        **kwargs : dict
            Extra parameters to be passed to the GATConv layer
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the
        self.feature_size = out_channels
        self.gatconv = GATConv(node_in, out_channels, **kwargs)
        self.bn = norm_factory(normalization, out_channels)
        self.act = act_factory(activation)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        *args: object,
    ) -> torch.Tensor:
        """Pass a batch of node/edges through the edge update layer.

        Parameters
        ----------
        node_feats : torch.Tensor
            (C, N_c) Node features
        edge_index : torch.Tensor
            (2, E) Incidence matrix
        *args : list, optional
            Other parameters passed but not needed

        Returns
        -------
        torch.Tensor
            (C, N_o) Updated node features
        """
        x = self.gatconv(node_feats, edge_index)
        x = self.bn(x)
        x = self.act(x)

        return x
