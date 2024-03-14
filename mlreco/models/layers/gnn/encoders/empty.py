"""Module which defines encoders that produce no features."""

import torch

__all__ = ['EmptyClusterNodeEncoder', 'EmptyClusterEdgeEncoder',
           'EmptyClusterGlobalEncoder']


class EmptyClusterNodeEncoder(torch.nn.Module):
    """Produces empty cluster node features."""
    name = 'empty'

    def forward(self, data, clusts, **kwargs):
        """Generate empty node features for one batch of data.

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
            (C, 0) Empty set of features per cluster
        """
        feats =  torch.empty((len(clusts.index_list), 0),
                           dtype=data.dtype, device=data.device)
        return TensorBatch(feats, clusts.counts)


class EmptyClusterEdgeEncoder(torch.nn.Module):
    """Produces empty cluster edge features."""
    name = 'empty'

    def forward(self, clusts, **kwargs):
        """Generate empty edge features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        edge_index : EdgeIndexBatch
            Incidence map between clusters
        **kwargs : dict, optional
            Additional objects no used by this encoder

        Returns
        -------
        TensorBatch
            (E, 0) Empty set of features per edge
        """
        feats = torch.empty((edge_index.full_index.shape[1], 0),
                           dtype=data.dtype, device=data.device)
        return TensorBatch(feats, edge_index.full_counts)


class EmptyClusterGlobalEncoder(torch.nn.Module):
    """Produces empty global graph features."""
    name = 'empty'

    def forward(self, clusts, **kwargs):
        """Generate empty global graph features for one batch of data.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional objects no used by this encoder

        Returns
        -------
        TensorBatch
            (B, 0) Empty set of features per batch entry
        """
        feats = torch.empty((data.batch_size, 0), 
                            dtype=data.dtype, device=data.device)
        return TensorBatch(
                feats, torch.ones(data.batch_size, dtype=torch.long))
