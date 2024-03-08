"""Module which defines encoders using convolutional neural networks."""

import torch
from typing import List

from mlreco.models.layers.cnn.cnn_encoder import SparseResidualEncoder

__all__ = ['ClustCNNNodeEncoder', 'ClustCNNEdgeEncoder']


class ClustCNNNodeEncoder(torch.nn.Module):
    """Produces node features using a sparse residual CNN encoder."""
    self.name = 'cnn'

    def __init__(self, **cfg):
        """Initializes the CNN-based node encoder.

        Simply passes the configuration along to the underlying sparse residual
        CNN encoder defined in :class:`SparseResidualEncoder`.

        Parameters
        ----------
        **cfg : dict, optional
            Configuration to pass along to the sparse residual encoder
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the sparse residual encoder
        self.encoder = SparseResidualEncoder(**encoder_cfg)

    def forward(self, data, clusts):
        """Generate node features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexListBatch
            (C) List of list of indexes that make each cluster
        
        Returns
        -------
        TensorBatch
           (C, N_e) Set of N_e features per cluster
        """
        # Use cluster ID as a batch ID, pass through CNN
        num_voxels = len(clusts.index)
        shape = (num_voxels, data.tensor.shape[1])
        cnn_data = torch.empty(shape, dtype=data.dtype, device=data.device)
        cnn_data = 
        for i, c in enumerate(clusts):
            cnn_data
            cnn_data = torch.cat((cnn_data, data[c,:5].float()))
            cnn_data[-len(c):,0] = i*torch.ones(len(c)).to(device)

        return self.encoder(cnn_data)


class ClustCNNEdgeEncoder(torch.nn.Module):
    """
    Uses a CNN to produce node features for cluster GNN

    """
    self.name = 'cnn'

    def __init__(self, model_config, **kwargs):
        super(ClustCNNEdgeEncoder, self).__init__()
        # Initialize the CNN
        self.encoder = SparseResidualEncoder(model_config)

    def forward(self, data, clusts, edge_index):

        # Check if the graph is undirected, select the relevant part of the edge index
        half_idx = int(edge_index.shape[1]/2)
        undirected = not edge_index.shape[1] or (not edge_index.shape[1]%2 and [edge_index[1,0], edge_index[0,0]] == edge_index[:,half_idx].tolist())
        if undirected: edge_index = edge_index[:,:half_idx]
.cnn import ClustCNN:
        # Use edge ID as a batch ID, pass through CNN
        device = data.device
        cnn_data = torch.empty((0, 5), device=device, dtype=torch.float)
        for i, e in enumerate(edge_index.T):
            ci, cj = clusts[e[0]], clusts[e[1]]
            cnn_data = torch.cat((cnn_data, data[ci,:5].float()))
            cnn_data = torch.cat((cnn_data, data[cj,:5].float()))
            cnn_data[-len(ci)-len(cj):,0] = i*torch.ones(len(ci)+len(cj)).to(device)

        feats = self.encoder(cnn_data)

        # If the graph is undirected, duplicate features
        if undirected:
            feats = torch.cat([feats,feats])

        return feats
