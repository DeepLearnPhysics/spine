"""Encoder adapters for objectized image-model inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from spine.data import TensorBatch

from ..cnn.encoder import SparseResidualEncoder

__all__ = [
    "ImageEncoder",
    "ImageCNNEncoder",
    "ImagePointNetEncoder",
    "image_encoder_factory",
]


class ImageEncoder(torch.nn.Module, ABC):
    """Common interface for encoders that produce one vector per object."""

    feature_size: int

    @abstractmethod
    def forward(self, data: TensorBatch) -> torch.Tensor:
        """Encode objectized coordinate-feature rows."""


class ImageCNNEncoder(ImageEncoder):
    """Adapt the pooled sparse residual CNN to the image-model interface."""

    name = "cnn"

    def __init__(self, **cfg: Any) -> None:
        """Initialize a sparse residual image encoder.

        Parameters
        ----------
        **cfg : dict
            Configuration forwarded to :class:`SparseResidualEncoder`.
        """
        # Initialize the parent class and wrapped sparse encoder
        super().__init__()
        self.encoder = SparseResidualEncoder(**cfg)
        self.feature_size = self.encoder.feature_size

    def forward(self, data: TensorBatch) -> torch.Tensor:
        """Encode one sparse image per entry in ``data``."""
        return self.encoder(data.torch_tensor())


class ImagePointNetEncoder(ImageEncoder):
    """Adapt PointNet++ to objectized SPINE coordinate-feature tables."""

    name = "pointnet"

    def __init__(
        self,
        data_dim: int = 3,
        num_input: int = 1,
        **cfg: Any,
    ) -> None:
        """Initialize a PointNet++ image encoder.

        Parameters
        ----------
        data_dim : int, default 3
            Number of spatial coordinate dimensions. PointNet currently
            requires three.
        num_input : int, default 1
            Number of point features following the coordinates.
        **cfg : dict
            PointNet++ architecture configuration.
        """
        # Initialize the parent class and validate the point-table layout
        super().__init__()
        if data_dim != 3:
            raise ValueError("The PointNet image encoder currently requires 3D data.")
        if num_input < 1:
            raise ValueError("`num_input` must be positive.")

        from ..pointcloud import PointNetEncoder

        # Initialize the wrapped PointNet implementation
        self.dimension = data_dim
        self.num_input = num_input
        self.encoder = PointNetEncoder({"pointnet": cfg})
        self.feature_size = self.encoder.feature_size

    def forward(self, data: TensorBatch) -> torch.Tensor:
        """Convert a SPINE table to a PyG batch and encode each object."""
        from torch_geometric.data import Data

        # Split the SPINE coordinate-feature table into PyG components
        table = data.torch_tensor()
        positions = table[:, 1 : self.dimension + 1]
        features = table[
            :,
            self.dimension + 1 : self.dimension + 1 + self.num_input,
        ]
        batch_ids = table[:, 0].long()

        # Encode the objectized point-cloud batch
        point_cloud = Data(x=features, pos=positions, batch=batch_ids)
        return self.encoder(point_cloud)


def image_encoder_factory(cfg: dict[str, Any]) -> ImageEncoder:
    """Instantiate an image encoder from a named configuration.

    Parameters
    ----------
    cfg : dict
        Encoder configuration containing ``name``.

    Returns
    -------
    ImageEncoder
        Encoder normalized to the objectized image interface.
    """
    # Extract the encoder name without mutating the caller's configuration
    config = dict(cfg)
    try:
        name = config.pop("name")
    except KeyError as err:
        raise ValueError("Image encoder configuration requires `name`.") from err

    # Resolve and instantiate the requested adapter
    encoders: dict[str, type[ImageEncoder]] = {
        "cnn": ImageCNNEncoder,
        "pointnet": ImagePointNetEncoder,
    }
    try:
        encoder_class = encoders[name]
    except KeyError as err:
        valid = ", ".join(sorted(encoders))
        raise ValueError(
            f"Unknown image encoder `{name}`. Choose from {valid}."
        ) from err
    return encoder_class(**config)
