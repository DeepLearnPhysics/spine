"""Module which defines a graph node feature update based on EdgeConv."""

from torch import nn

from torch_geometric.nn import EdgeConv

from spine.model.layer.common.mlp import MLP

__all__ = ['EConvNodeLayer']


class EConvNodeLayer(nn.Module):
    """EdgeConv module for extracting graph node features.

    This model first aggregates the feature vector (N_c) of the node being
    updated with a summary statistic of the difference between the features
    of the node and those of other connected nodes to form a (2*N_c) feature
    vector. This feature vector is then passed through a multi-layer
    perceptron (MLP) and outputs a (C, N_o) vector, with N_o the width
    of the unedrlying MLP (feature size of the hidden representation).

    Source: https://arxiv.org/abs/1801.07829
    """

    # Name of the node layer (as specified in the configuration)
    name = 'econv'

    def __init__(self, node_in, edge_in, glob_in, mlp, aggr='max', **kwargs):
        """Initialize the MLPs which are used to update the node features.

        Parameters
        ----------
        node_in : int
            Number of input node features
        edge_in : int
            Number of input edge features
        glob_in : int
            Number of input global features for the graph
        mlp : dict
            Configuration of the node update MLP
        aggr : str, default 'max'
            Node feature aggregation method
        **kwargs : dict
            Extra parameters to be passed to the EdgeConv layer
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the underlying MLP
        mlp = MLP(2*node_in, **mlp)
        self.feature_size = mlp.feature_size

        # Initialize the layer
        self.edgeconv = EdgeConv(nn=mlp, aggr=aggr, **kwargs)

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
        return self.edgeconv(node_feats, edge_index)
