"""Module which defines encoders using convolutional neural networks."""

import torch

from spine.model.layers.cnn.cnn_encoder import SparseResidualEncoder

from spine import TensorBatch, IndexBatch
from spine.utils.globals import BATCH_COL

__all__ = ['ClustCNNNodeEncoder', 'ClustCNNEdgeEncoder']


class ClustCNNNodeEncoder(torch.nn.Module):
    """Produces cluster node features using a sparse residual CNN encoder."""
    name = 'cnn'

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
        self.encoder = SparseResidualEncoder(**cfg)
        self.feature_size = self.encoder.feature_size

    def forward(self, data, clusts, **kwargs):
        """Generate CNN cluster node features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            Indexes that make up each cluster
        **kwargs : dict, optional
            Additional objects no used by this encoder
        
        Returns
        -------
        TensorBatch
           (C, N_c) Set of N_c features per cluster
        """
        # Use cluster ID as a batch ID, pass through CNN
        full_index = clusts.full_index
        num_voxels = len(full_index)
        shape = (num_voxels, data.tensor.shape[1])
        cnn_data = data.tensor[full_index].clone()
        cnn_data[:, BATCH_COL] = clusts.full_batch_ids

        # Pass the batched input through the encoder
        feats = self.encoder(cnn_data)

        return TensorBatch(feats, clusts.counts)


class ClustCNNEdgeEncoder(torch.nn.Module):
    """Produces cluster edge features using a sparse residual CNN encoder.

    Considers an edge as an image containing both ojbects connected by
    the edge in a single image.
    """
    name = 'cnn'

    def __init__(self, **cfg):
        """Initializes the CNN-based edge encoder.

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
        self.encoder = SparseResidualEncoder(**cfg)
        self.feature_size = self.encoder.feature_size

    def forward(self, data, clusts, edge_index, **kwargs):
        """Generate CNN cluster edge features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            Indexes that make up each cluster
        edge_index : EdgeIndexBatch
            Incidence map between clusters
        **kwargs : dict, optional
            Additional objects no used by this encoder
        
        Returns
        -------
        TensorBatch
           (C, N_e) Set of N_e features per edge
        """
        # Use edge ID as a batch ID, pass through CNN. For undirected graph,
        # only do it on half of the edges to save time (same features).
        cnn_data = []
        for i, e in enumerate(edge_index.directed_index):
            ci, cj = clusts.data[e[0]], clusts.data[e[1]]
            edge_data = torch.cat((data.tensor[ci], data.tensor[cj]))
            edge_data[:, BATCH_COL] = i
            cnn_data.append(edge_data)

        # Pass through the network
        if len(cnn_data):
            feats = self.encoder(torch.cat(cnn_data))
        else:
            feats = torch.empty((0, self.feature_size),
                                dtype=data.tensor.dtype,
                                device=data.tensor.device)
        
        # If the graph is undirected, add reciprocal features
        if not edge_index.directed:
            full_feats = torch.empty(
                    (2*feats.shape[0], feats.shape[1]),
                    dtype=feats.dtype, device=feats.device)
            full_feats[::2] = feats
            full_feats[1::2] = feats

            feats = full_feats

        return TensorBatch(feats, edge_index.counts)
