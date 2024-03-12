"""Defines the Metalayer message passing modeet"""

import torch
from torch import nn

from torch_geometric.nn import MetaLayer
from torch_geometric.utils import softmax

from .act_norm import act_construct, norm_construct


class MetaLayerModel(torch.nn.Module):
    """MetaLayer GNN Module for extracting node/edge/global features."""
    name = 'meta'

    def __init__(self, node_feats, node_output_feats, node_classes,
                 edge_feats, edge_output_feats, edge_classes,
                 global_feats=None, global_output_feats=None,
                 global_classes=None, reduction='mean', attention=False,
                 num_mp=3, activation={'name':'lrelu', 'negative_slope':0.1},
                 normalization='batch_norm', apply_softplus=False,
                 softplus_shift=1e-4):
        """Initializes the message passing block.

        Parameters
        ----------
        node_feats : int
            Number of node features
        node_output_feats : int
            Number of features in the hidden representation of the node
        node_classes : int
            Number of categories to classify the nodes into
        edge_feats : int
            Number of edge features
        edge_output_feats : int
            Number of features in the hidden representation of the edge
        edge_classes : int
            Number of categories to classify the global state into
        global_feats : int, optional
            Number of global features
        global_output_feats : int, optional
            Number of features in the hidden representation of the global state
        global_classes : int, optional
            Number of categories to classify the edges into
        num_mp : int, default 3
            Number of message passing steps
        reduction : str, default 'mean'
            Method for node features aggregation (one of 'mean', 'max' or 'min')
        attention : bool, default False
            Wether to use features to gate the messages of each edge in the
            node aggregation process of the message passing.
        activation : dict, default {'name':'lrelu', 'negative_slope':0.1}
            Activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            Normalization function configuration
        apply_softplus : bool, default False
            Whether to apply softplus on the final node and edge predictions
        softplus_shift : float, default 1-e4
            Shift to apply to the softplus output
        """
        # Initialize the parent class
        super().__init__()

        # Store the attributes
        self.node_feats     = node_feats
        self.node_output    = node_output_feats
        self.node_classes   = node_classes
        self.edge_feats     = edge_feats
        self.edge_output    = edge_output_feats
        self.edge_classes   = edge_classes
        self.global_feats   = global_feats
        self.global_output  = global_output_feats
        self.global_classes = global_classes

        self.num_mp    = num_mp
        self.reduction = reduction
        self.attention = attention
        self.act_cfg   = activation
        self.norm_cfg  = normalization

        # Initialize a softplus layer if need be
        self.apply_softplus = apply_softplus
        if apply_softplus:
            self.softplus = nn.Softplus()
            self.softplus_offset = softplus_offset

        # Loop over the number of message passing steps, initialize the
        # metalayer which updates the features it at each step
        self.edge_updates = nn.ModuleList()
        self.bn_node = nn.ModuleList() # REDUNDANT, REMOVE
        self.bn_edge = norm_construct(normalization, self.edge_feats) # DOESN'T DO SHIT
        node_input, edge_input = self.node_feats, self.edge_feats
        for i in range(self.num_mp):
            # MUST GO
            #self.bn_node.append(BatchNorm1d(node_input)) # HOW IS THIS DIFFERENT FROM BATCHNORM1D??? IT ISN'T
            self.bn_node.append(norm_construct(normalization, node_input)) 

            # Initialize the layer
            self.edge_updates.append(
                MetaLayer(
                    edge_model=EdgeLayer(
                        node_input, edge_input, self.edge_output,
                        activation, normalization),
                    node_model=NodeLayer(
                        node_input, self.edge_output, self.node_output,
                        activation, reduction, reduction, attention),
                    #global_model=GlobalModel(
                    #    node_output, node_output, global_output)
                    )
                )

            # Update the current number of node/edge features
            node_input = self.node_output
            edge_input = self.edge_output

        self.node_predictor = nn.Linear(self.node_output, self.node_classes)
        self.edge_predictor = nn.Linear(self.edge_output, self.edge_classes)

    def forward(self, node_features, edge_indices, edge_features, xbatch):

        x = node_features.view(-1, self.node_input)
        e = edge_features.view(-1, self.edge_input)

        for i in range(self.num_mp):
            x = self.bn_node[i](x)
            # add u and batch arguments for not having error in some old version
            x, e, _ = self.edge_updates[i](x, edge_indices, e, u=None, batch=xbatch)
        # print(edge_indices.shape)
        x_pred = self.node_predictor(x)
        e_pred = self.edge_predictor(e)

        if self.logit_mode == 'evidential':
            node_pred = self.softplus(x_pred) + self.softplus_shift
            edge_pred = self.softplus(e_pred) + self.softplus_shift
        else:
            node_pred = x_pred
            edge_pred = e_pred
        res = {
            'node_pred': [node_pred],
            'edge_pred': [edge_pred],
            'node_features': [x]
            }
        return res


class EdgeLayer(nn.Module):
    """
    An EdgeModel for predicting edge features.

    Example: Parent-Child Edge prediction and EM primary assignment prediction.

    INPUTS:

        DEFINITIONS:
            E: number of edges
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output edge features
            B: number of graphs (same as batch size)

        If an entry i->j is an edge, then we have source node feature
        F^i_x, target node feature F^j_x, and edge features F_e.

        - source: [E, F_x] Tensor, where E is the number of edges

        - target: [E, F_x] Tensor, where E is the number of edges

        - edge_attr: [E, F_e] Tensor, indicating input edge features.

        - global_features: [B, F_u] Tensor, where B is the number of graphs
        (equivalent to number of batches).

        - batch: [E] Tensor containing batch indices for each edge from 0 to B-1.

    RETURNS:

        - output: [E, F_o] Tensor with F_o output edge features.
    """
    def __init__(self, node_in, edge_in, edge_out, activation, normalization):
        super(EdgeLayer, self).__init__()
        # TODO: Construct Edge MLP
        leakiness = activation['negative_slope']# TMP TMP
        self.edge_mlp = nn.Sequential(
            nn.BatchNorm1d(2 * node_in + edge_in),
            nn.Linear(2 * node_in + edge_in, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(edge_out),
            nn.Linear(edge_out, edge_out)
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeLayer(nn.Module):
    """
    NodeModel for node feature prediction.

    Example: Particle Classification using node-level features.

    INPUTS:

        DEFINITIONS:
            N: number of nodes
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output node features
            B: number of graphs (same as batch size)

        If an entry i->j is an edge, then we have source node feature
        F^i_x, target node feature F^j_x, and edge features F_e.

        - source: [E, F_x] Tensor, where E is the number of edges

        - target: [E, F_x] Tensor, where E is the number of edges

        - edge_attr: [E, F_e] Tensor, indicating input edge features.

        - global_features: [B, F_u] Tensor, where B is the number of graphs
        (equivalent to number of batches).

        - batch: [E] Tensor containing batch indices for each edge from 0 to B-1.

    RETURNS:

        - output: [C, F_o] Tensor with F_o output node feature
    """
    def __init__(self, node_in, edge_in, node_out, activation, normalization, reduction, attention):
        super(NodeLayer, self).__init__()
        leakiness = activation['negative_slope']# TMP TMP

        self.node_mlp_1 = nn.Sequential(
            nn.BatchNorm1d(node_in + edge_in),
            nn.Linear(node_in + edge_in, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

        self.reduction = reduction

        self.node_mlp_2 = nn.Sequential(
            nn.BatchNorm1d(node_in + node_out),
            nn.Linear(node_in + node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(node_out),
            nn.Linear(node_out, node_out)
        )

        self.attention = attention
        if self.attention:
            self.gate = nn.Linear(node_out, 1)

    def forward(self, x, edge_index, edge_attr, u, batch):
        from torch_scatter import scatter
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        if self.attention:
            weights = softmax(self.gate(out), index=col)
            out = weights*out 
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce=self.reduction)
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(nn.Module):
    """
    Global Model for global feature prediction.

    Example: event classification (graph classification) over the whole image
    within a batch.

    Do Hierarchical Pooling to reduce features

    INPUTS:

        DEFINITIONS:
            N: number of nodes
            F_x: number of node features
            F_e: number of edge features
            F_u: number of global features
            F_o: number of output node features
            B: number of graphs (same as batch size)

        If an entry i->j is an edge, then we have source node feature
        F^i_x, target node feature F^j_x, and edge features F_e.

        - source: [E, F_x] Tensor, where E is the number of edges

        - target: [E, F_x] Tensor, where E is the number of edges

        - edge_attr: [E, F_e] Tensor, indicating input edge features.

        - global_features: [B, F_u] Tensor, where B is the number of graphs
        (equivalent to number of batches).

        - batch: [E] Tensor containing batch indices for each edge from 0 to B-1.

    RETURNS:

        - output: [C, F_o] Tensor with F_o output node feature
    """
    def __init__(self, node_in, batch_size, global_out, leakiness=0.0, reduction='mean'):
        super(GlobalModel, self).__init__()

        self.global_mlp = nn.Sequential(
            nn.BatchNorm1d(node_in + batch_size),
            nn.Linear(node_in + batch_size, global_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(global_out),
            nn.Linear(global_out, global_out),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.BatchNorm1d(global_out),
            nn.Linear(global_out, global_out)
        )

        self.reduction = reduction

    def forward(self, x, edge_index, edge_attr, u, batch):
        from torch_scatter import scatter
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter(x, batch, dim=0, reduce=self.reduction)], dim=1)
        return self.global_mlp(out)
