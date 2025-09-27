"""Module which defines encoders that mix geometric and CNN features together.

See :mod:`spine.model.layer.gnn.encode.geometric` and
:mod:`spine.model.layer.gnn.encode.cnn` for more information.
"""

import torch

from spine.model.layer.common.act_norm import act_factory

from .cnn import ClustCNNEdgeEncoder, ClustCNNNodeEncoder
from .geometric import ClustGeoEdgeEncoder, ClustGeoNodeEncoder

__all__ = ["ClustGeoCNNMixNodeEncoder", "ClustGeoCNNMixEdgeEncoder"]


class ClustGeoCNNMixNodeEncoder(torch.nn.Module):
    """Produces cluster node features using both geometric and CNN encoders."""

    # Name of the node encoder (as specified in the configuration)
    name = "geo_cnn_mix"

    def __init__(self, geo_encoder, cnn_encoder, activation="elu"):
        """Initialize the mixed encoder.

        Initializes the two underlying encoders:
        - :class:`ClustGeoNodeEncoder`
        - :class:`ClustCNNNodeEncoder`

        Parameters
        ----------
        geo_encoder : dict
            Geometric node encoder configuration
        cnn_encoder : dict,
            CNN node encoder configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the two underlying encoder
        self.geo_encoder = ClustGeoNodeEncoder(**geo_encoder)
        self.cnn_encoder = ClustCNNNodeEncoder(**cnn_encoder)

        # Get the number of features coming out of the encoder
        num_geo = self.geo_encoder.feature_size
        num_cnn = self.cnn_encoder.feature_size
        self.feature_size = num_geo + num_cnn

        # Initialize the final activation/linear layer
        self.act = act_factory(activation)
        self.linear = torch.nn.Linear(self.features_size, self.feature_size)

    def forward(self, data, clusts, **kwargs):
        """Generate mixed cluster node features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            (C) List of list of indexes that make up each cluster
        **kwargs : dict, optional
            Additional objects no used by this encoder

        Returns
        -------
        TensorBatch
            (C, N_c) Set of N_c features per cluster
        """
        # Get the features
        features_geo = self.geo_encoder(data, clusts, **kwargs).tensor
        features_cnn = self.cnn_encoder(data, clusts, **kwargs).tensor
        features_mix = torch.cat([features_geo, features_cnn], dim=1)

        # Push features through the final activation/linear layer
        result = self.act(features_mix)
        result = self.linear(result)

        return TensorBatch(result, clusts.list_counts)


class ClustGeoCNNMixEdgeEncoder(torch.nn.Module):
    """Produces cluster edge features using both geometric and CNN encoders."""

    # Name of the edge encoder (as specified in the configuration)
    name = "geo_cnn_mix"

    def __init__(self, geo_encoder, cnn_encoder):
        """Initialize the mixed encoder.

        Initializes the two underlying encoders:
        - :class:`ClustGeoEdgeEncoder``
        - :class:`ClustCNNEdgeEncoder`

        Parameters
        ----------
        geo_encoder : dict
            Geometric edge encoder configuration
        cnn_encoder : dict,
            CNN edge encoder configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the two underlying encoder
        self.geo_encoder = ClustGeoEdgeEncoder(**geo_encoder)
        self.cnn_encoder = ClustCNNEdgeEncoder(**cnn_encoder)

        # Get the number of features coming out of the encoder
        num_geo = self.geo_encoder.feature_size
        num_cnn = self.cnn_encoder.feature_size
        self.feature_size = num_geo + num_cnn

        # Initialize the final activation/linear layer
        self.act = act_factory(activation)
        self.linear = torch.nn.Linear(self.features_size, self.feature_size)

    def forward(self, data, clusts, edge_index, **kwargs):
        """Generate mixed cluster edge features for one batch of data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) Batch of sparse tensors
        clusts : IndexBatch
            (C) List of list of indexes that make up each cluster
        edge_index : EdgeIndexBatch
            Incidence map between clusters
        **kwargs : dict, optional
            Additional objects no used by this encoder

        Returns
        -------
        TensorBatch
            (C, N_e) Set of N_e features per edge
        """
        # Get the features
        features_geo = self.geo_encoder(data, clusts, **kwargs).tensor
        features_cnn = self.cnn_encoder(data, clusts, **kwargs).tensor
        features_mix = torch.cat([features_geo, features_cnn], dim=1)

        # Push features through the final activation/linear layer
        result = self.act(features_mix)
        result = self.linear(result)

        return TensorBatch(result, edge_index.counts)
