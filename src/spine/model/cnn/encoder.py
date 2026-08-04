"""Defines CNN encoder backbones for image feature extraction."""

from __future__ import annotations

from typing import Any

import torch

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
            Whether to append normalized spatial coordinates to the configured
            raw input features. Coordinate channels are added automatically and
            should not be included in ``num_input``.
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
        # Coordinate convolution adds spatial positions to the configured raw
        # input features. Size the inherited input layer accordingly while
        # retaining the source feature count for table slicing.
        encoder_cfg = dict(cfg)
        self.input_features = int(encoder_cfg.get("num_input", 1))
        if coord_conv:
            dimension = int(encoder_cfg.get("data_dim", 3))
            encoder_cfg["num_input"] = self.input_features + dimension

        # Initialize the parent class
        super().__init__(encoder_cfg)

        # Store attributes
        self.coord_conv = coord_conv
        self.pool_mode = pool_mode

        # Initialize the final pooling layer
        if self.spatial_size is None and (coord_conv or pool_mode == "conv"):
            raise ValueError(
                "`spatial_size` is required for coordinate convolution or "
                "convolutional pooling."
            )

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
            spatial_size = self.spatial_size
            if spatial_size is None:  # Constructor validation narrows the type.
                raise RuntimeError("Convolutional pooling requires `spatial_size`.")
            final_tensor_shape = spatial_size // (2 ** (self.depth - 1))
            self.pool = torch.nn.Sequential(
                sparse.Convolution(
                    in_channels=self.num_planes[-1],
                    out_channels=self.num_planes[-1],
                    kernel_size=final_tensor_shape,
                    stride=final_tensor_shape,
                    dimension=self.dimension,
                ),
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
        # Build coordinate and input feature tensors
        coords = x[:, : self.dimension + 1]
        features = x[
            :,
            self.dimension + 1 : self.dimension + 1 + self.input_features,
        ]

        # If requested, append the normalized coordinates to the feature tensor
        if self.coord_conv:
            spatial_size = self.spatial_size
            if spatial_size is None:  # Constructor validation narrows the type.
                raise RuntimeError("Coordinate convolution requires `spatial_size`.")
            normalized_coords = x[:, 1 : self.dimension + 1] / spatial_size
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
