"""Module which defines encoders that produce no features."""

import torch

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch

__all__ = [
    "EmptyClusterNodeEncoder",
    "EmptyClusterEdgeEncoder",
    "EmptyClusterGlobalEncoder",
]


class EmptyClusterNodeEncoder(torch.nn.Module):
    """Produces empty cluster node features."""

    # Name of the node encoder (as specified in the configuration)
    name = "empty"

    feature_size = 0

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        **kwargs: object,
    ) -> TensorBatch:
        """Generate empty node features for one batch of data.

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
            (C, 0) Empty set of features per cluster
        """
        feats = torch.empty(
            (len(clusts.index_list), 0), dtype=data.dtype, device=data.device
        )

        return TensorBatch(feats, clusts.counts)


class EmptyClusterEdgeEncoder(torch.nn.Module):
    """Produces empty cluster edge features."""

    # Name of the edge encoder (as specified in the configuration)
    name = "empty"

    feature_size = 0

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        **kwargs: object,
    ) -> TensorBatch:
        """Generate empty edge features for one batch of data.

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
            (E, 0) Empty set of features per edge
        """
        feats = torch.empty(
            (edge_index.index.shape[1], 0), dtype=data.dtype, device=data.device
        )

        return TensorBatch(feats, edge_index.counts)


class EmptyClusterGlobalEncoder(torch.nn.Module):
    """Produces empty global graph features."""

    # Name of the global encoder (as specified in the configuration)
    name = "empty"

    feature_size = 0

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        **kwargs: object,
    ) -> TensorBatch:
        """Generate empty global graph features for one batch of data.

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
            (B, 0) Empty set of features per batch entry
        """
        feats = torch.empty((data.batch_size, 0), dtype=data.dtype, device=data.device)

        counts = torch.ones(
            data.batch_size,
            dtype=torch.long,
            device=data.device,
        )
        return TensorBatch(feats, counts)
