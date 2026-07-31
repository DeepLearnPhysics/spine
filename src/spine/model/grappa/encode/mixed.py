"""Encoders that combine geometric and convolutional graph features."""

from __future__ import annotations

from typing import Any

import torch

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.model.common.act_norm import act_factory

from .cnn import ClustCNNEdgeEncoder, ClustCNNNodeEncoder
from .geometric import ClustGeoEdgeEncoder, ClustGeoNodeEncoder

__all__ = ["ClustGeoCNNMixNodeEncoder", "ClustGeoCNNMixEdgeEncoder"]


class ClustGeoCNNMixNodeEncoder(torch.nn.Module):
    """Combine geometric and sparse-CNN features for each graph node."""

    name = "geo_cnn_mix"

    def __init__(
        self,
        geo_encoder: dict[str, Any],
        cnn_encoder: dict[str, Any],
        activation: str | dict[str, Any] = "elu",
    ) -> None:
        """Initialize the mixed node encoder.

        Parameters
        ----------
        geo_encoder : dict
            Geometric node-encoder configuration.
        cnn_encoder : dict
            Sparse-CNN node-encoder configuration.
        activation : str or dict, default "elu"
            Activation applied before the feature-mixing projection.
        """
        super().__init__()

        self.geo_encoder = ClustGeoNodeEncoder(**geo_encoder)
        self.cnn_encoder = ClustCNNNodeEncoder(**cnn_encoder)

        self.feature_size = (
            self.geo_encoder.feature_size + self.cnn_encoder.feature_size
        )
        self.act = act_factory(activation)
        self.linear = torch.nn.Linear(self.feature_size, self.feature_size)

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        **kwargs: object,
    ) -> TensorBatch:
        """Generate mixed node features.

        Parameters
        ----------
        data : TensorBatch
            Batched voxel/value table.
        clusts : IndexBatch
            Batched cluster index list.
        **kwargs : object
            Additional inputs forwarded to the component encoders.

        Returns
        -------
        TensorBatch
            One mixed feature vector per cluster.

        Raises
        ------
        ValueError
            If the geometric encoder is configured to return auxiliary points.
        """
        geometric_output = self.geo_encoder(data, clusts, **kwargs)
        if isinstance(geometric_output, tuple):
            raise ValueError(
                "Mixed node encoding does not support returning auxiliary points."
            )
        cnn_features = self.cnn_encoder(data, clusts, **kwargs).torch_tensor()
        geometric_features = geometric_output.to_tensor(
            dtype=cnn_features.dtype,
            device=cnn_features.device,
        ).torch_tensor()
        mixed_features = torch.cat((geometric_features, cnn_features), dim=1)

        output = self.linear(self.act(mixed_features))
        return TensorBatch(output, clusts.counts)


class ClustGeoCNNMixEdgeEncoder(torch.nn.Module):
    """Combine geometric and sparse-CNN features for each graph edge."""

    name = "geo_cnn_mix"

    def __init__(
        self,
        geo_encoder: dict[str, Any],
        cnn_encoder: dict[str, Any],
        activation: str | dict[str, Any] = "elu",
    ) -> None:
        """Initialize the mixed edge encoder.

        Parameters
        ----------
        geo_encoder : dict
            Geometric edge-encoder configuration.
        cnn_encoder : dict
            Sparse-CNN edge-encoder configuration.
        activation : str or dict, default "elu"
            Activation applied before the feature-mixing projection.
        """
        super().__init__()

        self.geo_encoder = ClustGeoEdgeEncoder(**geo_encoder)
        self.cnn_encoder = ClustCNNEdgeEncoder(**cnn_encoder)

        self.feature_size = (
            self.geo_encoder.feature_size + self.cnn_encoder.feature_size
        )
        self.act = act_factory(activation)
        self.linear = torch.nn.Linear(self.feature_size, self.feature_size)

    def forward(
        self,
        data: TensorBatch,
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        **kwargs: object,
    ) -> TensorBatch:
        """Generate mixed edge features.

        Parameters
        ----------
        data : TensorBatch
            Batched voxel/value table.
        clusts : IndexBatch
            Batched cluster index list.
        edge_index : EdgeIndexBatch
            Batched graph incidence map.
        **kwargs : object
            Additional inputs forwarded to the component encoders.

        Returns
        -------
        TensorBatch
            One mixed feature vector per edge.
        """
        geometric_output = self.geo_encoder(
            data,
            clusts,
            edge_index,
            **kwargs,
        )
        cnn_features = self.cnn_encoder(
            data,
            clusts,
            edge_index,
            **kwargs,
        ).torch_tensor()
        geometric_features = geometric_output.to_tensor(
            dtype=cnn_features.dtype,
            device=cnn_features.device,
        ).torch_tensor()
        mixed_features = torch.cat((geometric_features, cnn_features), dim=1)

        output = self.linear(self.act(mixed_features))
        return TensorBatch(output, edge_index.counts)
