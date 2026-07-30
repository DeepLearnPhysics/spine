"""Defines CNN encoder backbones for image feature extraction."""

from __future__ import annotations

from typing import Any

import torch

from spine.constants import COORD_COLS, VALUE_COL
from spine.model import sparse

from .uresnet_layers import UResNetEncoder

__all__ = ["SparseResidualEncoder"]


class SparseResidualEncoder(UResNetEncoder):
    """Reduce a sparse image to one fixed-width feature vector per batch.

    This class extends :class:`UResNetEncoder` with optional coordinate
    features, global or convolutional pooling, and a final linear projection.
    It is used by whole-image and graph-object feature encoders.
    """

    def __init__(
        self,
        coord_conv: bool = False,
        pool_mode: str = "avg",
        feature_size: int = 512,
        **cfg: Any,
    ) -> None:
        """Initialize the pooled sparse residual encoder.

        Passes most of the configuration along to the underlying sparse
        residual CNN encoder defined in :class:`UResNetEncoder`.

        Parameters
        ----------
        coord_conv : bool, default False
            Whether to append normalized spatial coordinates to the scalar
            input feature. If enabled, ``num_input`` in ``cfg`` must include
            these additional channels.
        pool_mode : {"avg", "sum", "max", "conv"}, default "avg"
            Operation used to reduce the deepest sparse feature plane.
        feature_size : int, default 512
            Width of the returned feature vectors.
        **cfg : dict, optional
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.

        Raises
        ------
        ValueError
            If ``spatial_size`` is missing or ``pool_mode`` is unknown.
        """
        # Initialize the parent class
        super().__init__(cfg)

        # Store attributes
        self.coord_conv = coord_conv
        self.pool_mode = pool_mode

        # Initialize the final pooling layer
        if self.spatial_size is None:
            raise ValueError(
                "Must specify `spatial_size` to determine the final pooling size."
            )
        final_tensor_shape = self.spatial_size // (2 ** (self.depth - 1))

        if pool_mode == "avg":
            # Average pooling
            self.pool = sparse.GlobalAvgPooling()

        elif pool_mode == "sum":
            # Sum pooling
            self.pool = sparse.GlobalSumPooling()

        elif pool_mode == "max":
            # Max pooling
            self.pool = sparse.GlobalMaxPooling()

        elif pool_mode == "conv":
            # Strided convolution
            self.pool = torch.nn.Sequential(
                sparse.Convolution(
                    in_channels=self.num_planes[-1],
                    out_channels=self.num_planes[-1],
                    kernel_size=final_tensor_shape,
                    stride=final_tensor_shape,
                    dimension=self.dimension,
                ),
                sparse.GlobalPooling(),
            )

        else:
            raise ValueError(
                f"Pooling mode not recognized: {self.pool_mode}. Must be "
                "one of 'avg', 'sum', 'max' or 'conv'"
            )

        # Initialize the final linear layer
        self.feature_size = feature_size
        self.linear = sparse.Linear(self.num_planes[-1], self.feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a coordinate-feature table into batch-level features.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, 1 + D + C)`` table containing batch IDs, coordinates and
            input features.

        Returns
        -------
        torch.Tensor
            ``(B, feature_size)`` matrix with one row per batch entry.
        """
        # Build an input feature tensor
        coords = x[:, :VALUE_COL]
        features = x[:, VALUE_COL].view(-1, 1)

        # If requested, append the normalized coordinates to the feature tensor
        if self.coord_conv:
            normalized_coords = x[:, COORD_COLS] / self.spatial_size
            features = torch.cat([normalized_coords, features], dim=1)

        # Build a sparse tensor
        x = sparse.SparseTensor(coordinates=coords.int(), features=features)

        # Pass through the CNN encoder
        output = super().forward(x)
        final_tensor = output["final_tensor"]

        # Pool the last layer
        pooled = self.pool(final_tensor)

        # Put it through a linear layer
        latent = self.linear(pooled)

        return latent.features
