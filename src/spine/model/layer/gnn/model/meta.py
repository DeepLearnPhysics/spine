"""Module which contains a generic GNN message passing implementation."""

from torch import nn
from torch_geometric.nn import MetaLayer

from spine.data import TensorBatch
from spine.model.layer.common.act_norm import norm_factory

from .factories import edge_layer_factory, global_layer_factory, node_layer_factory

__all__ = ["MetaLayerGNN"]


class MetaLayerGNN(nn.Module):
    """Completely generic message-passing GNN."""

    # Name of the model (as specified in the configuration)
    name = "meta"

    def __init__(
        self,
        node_feats=0,
        node_layer=None,
        node_pred=True,
        edge_feats=0,
        edge_layer=None,
        edge_pred=True,
        global_feats=0,
        global_layer=None,
        global_pred=True,
        num_mp=3,
        input_normalization="batch_norm",
    ):
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

        # Check that at least one of the output features is needed
        assert (
            node_pred or edge_pred or global_pred
        ), "Must request at least one type of GNN features to be output."

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
        self.mp_layers = nn.ModuleList()
        node_nf, edge_nf, glob_nf = (node_feats, edge_feats, global_feats)

        for l in range(self.num_mp):
            # Initialize the edge update layer
            edge_model = None
            if edge_layer is not None:
                edge_model = edge_layer_factory(edge_layer, node_nf, edge_nf, glob_nf)
                edge_nf = edge_model.feature_size

            # Initialize the node update layer
            node_model = None
            if node_layer is not None:
                if (node_pred or global_pred) or l < (self.num_mp - 1):
                    node_model = node_layer_factory(
                        node_layer, node_nf, edge_nf, glob_nf
                    )
                    node_nf = node_model.feature_size

            # Initialize the global update layer
            global_model = None
            if global_layer is not None:
                if global_pred or l < (self.num_mp - 1):
                    global_model = global_layer_factory(global_layer, node_nf, glob_nf)
                    glob_nf = global_model.feature_size

            # Build the complete metalayer
            self.mp_layers.append(MetaLayer(edge_model, node_model, global_model))

        # Store the feature size of each of the outputs
        self.node_feature_size = node_nf
        self.edge_feature_size = edge_nf
        self.global_feature_size = glob_nf

    def forward(self, node_feats, edge_index, edge_feats, glob_feats, batch):
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
        x, e, u = node_feats.tensor, None, None
        if self.node_bn is not None:
            x = self.node_bn(node_feats.tensor)
        if edge_feats is not None:
            e = edge_feats.tensor
            if self.edge_bn is not None:
                e = self.edge_bn(e)
        if glob_feats is not None:
            u = glob_feats.tensor
            if self.global_bn is not None:
                u = self.global_bn(u)

        # Loop over the message passing steps, update the graph features
        for l in range(self.num_mp):
            x, e, u = self.mp_layers[l](x, edge_index, e, u, batch)

        # Initialize and return result dictionary
        result = {}
        if self.mp_layers[0].node_model is not None and self.node_pred:
            result["node_features"] = TensorBatch(x, node_feats.counts)
        if self.mp_layers[0].edge_model is not None and self.edge_pred:
            result["edge_features"] = TensorBatch(e, edge_feats.counts)
        if self.mp_layers[0].global_model is not None and self.global_pred:
            result["global_features"] = TensorBatch(u, global_feats.counts)

        return result
