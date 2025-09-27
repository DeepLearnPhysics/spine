"""Module which defines a graph node feature update based on AGNNConv."""

from torch import nn
from torch_geometric.nn import AGNNConv

from spine.model.layer.common.act_norm import act_factory, norm_factory

__all__ = ["AGNNConvNodeLayer"]


class AGNNConvNodeLayer(nn.Module):
    """AGNNConv module for extracting graph node features.

    This model simply takes a simple attention-based convolution of a node
    with all of its neighbors to update the initial node feature vector (N_c),
    preserving the original size and returning an updated (N_c) vector.

    Source: https://arxiv.org/abs/1803.03735
    """

    # Name of the node layer (as specified in the configuration)
    name = "agnnconv"

    def __init__(self, node_in, edge_in, glob_in, activation, normalization, **kwargs):
        """Initialize the MLPs which are used to update the node features.

        Parameters
        ----------
        node_in : int
            Number of input node features
        edge_in : int
            Number of input edge features
        glob_in : int
            Number of input global features for the graph
        activation : Union[str, dict], default 'relu'
            Activation function configuration
        normalization : Union[str, dict], default 'batch_norm'
            Normalization function configuration
        **kwargs : dict
            Extra parameters to be passed to the AGNNConv layer
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the layer
        self.feature_size = node_in
        self.agnnconv = AGNNConv(**kwargs)
        self.bn = norm_factory(normalization, node_in)
        self.act = act_factory(activation)

    def forward(self, node_feats, edge_index, *args):
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
        x = self.agnnconv(node_feats, edge_index)
        x = self.bn(x)
        x = self.act(x)

        return x
