"""Module which defines generic graph features update based on MLPs."""

import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from spine.model.layer.common.mlp import MLP

__all__ = ["MLPEdgeLayer", "MLPNodeLayer", "MLPGlobalLayer"]


class MLPEdgeLayer(nn.Module):
    """Model used to update the set of edge features.

    For each edge, this model first aggregates the features from the source
    node (N_c) with those of the sink node (N_c) and those of the edge (N_e)
    and those of the graph (N_g) to form an input feature vector of dimension
    (E, 2*N_c + N_e + N_g). This feature vector is then passed through a
    multi-layer perceptron (MLP) and outputs an (E, N_o) vector, with N_o the
    width of the MLP (feature size of the hidden representation).
    """

    # Name of the edge layer (as specified in the configuration)
    name = "mlp"

    def __init__(self, node_in, edge_in, glob_in, mlp):
        """Initialize the MLP which is used to update the edge features.

        Parameters
        ----------
        edge_in : int
            Number of input edge features
        node_in : int
            Number of input node features
        glob_in : int
            Number of input global features for the graph
        mlp : dict
            Configuration of the edge update MLP
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the edge update MLP
        self.mlp = MLP(2 * node_in + edge_in + glob_in, **mlp)
        self.feature_size = self.mlp.feature_size

    def forward(self, src_feats, dest_feats, edge_feats, glob_feats, batch):
        """Pass a batch of node/edges through the edge update layer.

        Parameters
        ----------
        src_feats : torch.Tensor
            (E, N_c) Source node features
        dest_feats : torch.Tensor
            (E, N_c) Sink node features
        edge_feats : torch.Tensor
            (E, N_e) Edge features
        glob_feats : torch.Tensor
            (B, N_g) Global features
        batch : torch.Tensor
            (E) ID of the entry of each of the edges within the batch

        Returns
        -------
        torch.Tensor
            (C, N_o) Updated edge features
        """
        # Initialize the input to the edge update
        input_feats = torch.cat([src_feats, dest_feats, edge_feats], dim=1)
        if glob_feats is not None:
            input_feats = torch.cat([input_feats, glob_feats[batch]], dim=1)

        # Pass those features through the MLP
        return self.mlp(input_feats)


class MLPNodeLayer(nn.Module):
    """Model used to update the set of node features.

    For each node, this model proceeds in two seperate steps:
    - A message formation step
    - A message aggregation step

    For each edge, the message formation step consists in first aggregating
    the source node features (N_c) with the edge features (N_e) to form an
    input feature vector of dimension (N_c + N_e). This feature vector is
    then passed through a multi-layer perceptron (MLP) and outputs an (N_o)
    vector, with N_o the width of the MLP (feature size of the hidden
    representation). This feature vector is the message associated with that
    edge.

    For each node, the message aggregations step consists in taking a
    summary statistic of all edge features which correspond to edges that have
    that node as a sink. The so-formed feature vector of size (N_o) is then
    stacked with the node features (N_c) and the global graph features (N_g)
    to form a (N_o + N_c + N_g) feature vector. This new vector is passed
    through a second MLP to update the node features to (N_o').
    """

    # Name of the node layer (as specified in the configuration)
    name = "mlp"

    def __init__(
        self,
        node_in,
        edge_in,
        glob_in,
        message_mlp,
        aggr_mlp,
        reduction="mean",
        attention=False,
    ):
        """Initialize the MLPs which are used to update the node features.

        Parameters
        ----------
        node_in : int
            Number of input node features
        edge_in : int
            Number of input edge features
        glob_in : int
            Number of input global features for the graph
        message_mlp : dict
            Configuration of the message creation MLP
        aggr_mlp : dict
            Configuration of the message aggregtion MLP
        reduction : str, default 'mean'
            Message feature aggregation function
        attention : bool, default False
            Whether or not to learn explicit attention to particular messages
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the message creation MLP
        self.message_mlp = MLP(node_in + edge_in, **message_mlp)
        message_width = self.message_mlp.width[-1]

        # Initialize the aggregation MLP
        self.aggr_mlp = MLP(node_in + message_width + glob_in, **aggr_mlp)
        self.feature_size = self.aggr_mlp.feature_size

        # Store the edge feature aggregation mode
        self.reduction = reduction

        # If self-attention is needed, initialize a simple gate layer
        self.attention = attention
        if self.attention:
            self.gate = nn.Linear(node_out, 1)

    def forward(self, node_feats, edge_index, edge_feats, glob_feats, batch):
        """Pass a batch of node/edges through the edge update layer.

        Parameters
        ----------
        node_feats : torch.Tensor
            (C, N_c) Node features
        edge_index : torch.Tensor
            (2, E) Incidence matrix
        edge_feats : torch.Tensor
            (E, N_e) Edge features
        glob_feats : torch.Tensor
            (B, N_g) Global features
        batch : torch.Tensor
            (C) ID of the entry of each of the nodes within the batch

        Returns
        -------
        torch.Tensor
            (C, N_o') Updated node features
        """
        # Slip the edge index between source and sink
        src, dest = edge_index

        # Concatenate edge features with the source nodes, pass through MLP
        input_feats = torch.cat([node_feats[src], edge_feats], dim=1)
        message_feats = self.message_mlp(input_feats)

        # If requested, make the network learn to explicitely tune down
        # features associated with certaine edges
        if self.attention:
            # Compute a softmax score for nodes that share a sink
            weights = softmax(self.gate(message_feats), index=dest)

            # Apply the gate weigths to the message features
            message_feats = weights * message_feats

        # Aggregate messages to form one coherent message per node
        message_feats = scatter(
            message_feats,
            dest,
            dim=0,
            dim_size=node_feats.size(0),
            reduce=self.reduction,
        )

        # Aggerate the message with the node features, pass through MLP
        input_feats = torch.cat([node_feats, message_feats], dim=1)
        if glob_feats is not None:
            input_feats = torch.cat([input_feats, glob_feats[batch]], dim=1)

        return self.aggr_mlp(input_feats)


class MLPGlobalLayer(nn.Module):
    """Model used to update the set of global graph features.

    For each graph (one per entry in the batch of B graphs), this model first
    takes a summary statistic of the information from all the nodes in the
    graph (N_c). It then aggregates this feature vector with the one associated
    with the graph itself (N_g) to form a feature fector of dimension
    (N_c + N_g). This feature vector is then passed through a multi-layer
    perceptron (MLP) and outputs a (B, N_o) vector, with N_o the width of the
    MLP (feature size of the hidden representation).
    """

    # Name of the global layer (as specified in the configuration)
    name = "mlp"

    def __init__(self, node_in, glob_in, mlp, reduction="mean"):
        """Initialize the MLP which is used to update the edge features.

        Parameters
        ----------
        glob_in : int
            Number of input global features for the graph
        node_in : int
            Number of input node features
        mlp : dict
            Configuration of the global representation update MLP
        reduction : str, default 'mean'
            Message feature aggregation function
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the global representation update MLP
        self.mlp = MLP(node_in + glob_in, **mlp)
        self.feature_size = self.mlp.feature_size

        # Store the node feature aggregation mode
        self.reduction = reduction

    def forward(self, node_feats, edge_index, edge_feats, glob_feats, batch):
        """Pass a batch of node/edges through the global update layer.

        Parameters
        ----------
        node_feats : torch.Tensor
            (C, N_c) Source node features
        edge_index : torch.Tensor
            (2, E) Incidence matrix
        edge_feats : torch.Tensor
            (E, N_e) Edge features
        glob_feats : torch.Tensor
            (B, N_g) Global features
        batch : torch.Tensor
            (C) ID of the entry of each of the nodes within the batch

        Returns
        -------
        torch.Tensor
            (B, N_o) Updated global features
        """
        # Aggregate node features to form one single feature vector
        all_node_feats = scatter(node_feats, batch, dim=0, reduce=self.reduction)

        # Initialize the input to the global update
        input_feats = torch.cat([all_node_feats, glob_feats], dim=1)

        # Pass those features through the MLP
        return self.mlp(input_feats)
