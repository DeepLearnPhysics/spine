"""Module which defines encoders using convolutional neural networks."""

from typing import Any

import torch

from spine.constants import BATCH_COL
from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.model.layer.cnn.encoder import SparseResidualEncoder

__all__ = ["ClustCNNNodeEncoder", "ClustCNNEdgeEncoder", "ClustCNNGlobalEncoder"]


class ClustCNNNodeEncoder(torch.nn.Module):
    """Produces cluster node features using a sparse residual CNN encoder."""

    # Name of the node encoder (as specified in the configuration)
    name = "cnn"

    def __init__(self, **cfg: Any) -> None:
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

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        **kwargs: object,
    ) -> TensorBatch:
        """Generate CNN cluster node features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            Indexes that make up each cluster
        **kwargs : dict, optional
            Additional objects not used by this encoder

        Returns
        -------
        TensorBatch
            (C, N_c) Set of N_c features per cluster
        """
        # Use cluster ID as a batch ID, pass through CNN
        full_index = clusts.full_index
        cnn_data = data.torch_tensor()[full_index].clone()
        cnn_data[:, BATCH_COL] = torch.as_tensor(
            clusts.index_ids,
            device=cnn_data.device,
        )

        # Pass the batched input through the encoder
        feats = self.encoder(cnn_data)

        return TensorBatch(feats, clusts.counts)


class ClustCNNEdgeEncoder(torch.nn.Module):
    """Produces cluster edge features using a sparse residual CNN encoder.

    Considers an edge as an image containing both objects connected by
    the edge in a single image.
    """

    # Name of the edge encoder (as specified in the configuration)
    name = "cnn"

    def __init__(self, **cfg: Any) -> None:
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

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        **kwargs: object,
    ) -> TensorBatch:
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
            Additional objects not used by this encoder

        Returns
        -------
        TensorBatch
            (E, N_e) Set of N_e features per edge
        """
        # Use edge ID as a batch ID, pass through CNN. For undirected graph,
        # only do it on half of the edges to save time (same features).
        cnn_data = []
        data_tensor = data.torch_tensor()
        for edge_id, edge in enumerate(edge_index.directed_index_t):
            first_cluster = clusts.index_list[edge[0]]
            second_cluster = clusts.index_list[edge[1]]
            edge_data = torch.cat(
                (data_tensor[first_cluster], data_tensor[second_cluster])
            )
            edge_data[:, BATCH_COL] = edge_id
            cnn_data.append(edge_data)

        # Pass through the network
        if len(cnn_data) > 0:
            feats = self.encoder(torch.cat(cnn_data))

        else:
            feats = torch.empty(
                (0, self.feature_size),
                dtype=data_tensor.dtype,
                device=data_tensor.device,
            )

        # If the graph is undirected, add reciprocal features
        if not edge_index.directed:
            full_feats = torch.empty(
                (2 * feats.shape[0], feats.shape[1]),
                dtype=feats.dtype,
                device=feats.device,
            )
            full_feats[::2] = feats
            full_feats[1::2] = feats

            feats = full_feats

        return TensorBatch(feats, edge_index.counts)


class ClustCNNGlobalEncoder(torch.nn.Module):
    """Produces graph-wide features using a sparse residual CNN encoder.

    Considers the whole graph as an image containing all objects in it.
    """

    # Name of the global encoder (as specified in the configuration)
    name = "cnn"

    def __init__(self, **cfg: Any) -> None:
        """Initializes the CNN-based global encoder.

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

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        **kwargs: object,
    ) -> TensorBatch:
        """Generate CNN global graph features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            Indexes that make up each cluster
        **kwargs : dict, optional
            Additional objects not used by this encoder

        Returns
        -------
        TensorBatch
            (B, N_g) Set of N_g global graph features per batch entry
        """
        # Restrict the set of points to those in the graph clusters
        full_index = clusts.full_index
        cnn_data = data.torch_tensor()[full_index]

        # Pass the batched input through the encoder
        feats = self.encoder(cnn_data)

        return TensorBatch(feats, [1] * clusts.batch_size)
