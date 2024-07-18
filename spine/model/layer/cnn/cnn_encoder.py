"""Defines CNN encoder backbones for image feature extraction."""

import torch

import MinkowskiEngine as ME

from .uresnet_layers import UResNetEncoder

from spine.utils.globals import COORD_COLS, VALUE_COL

__all__ = ['SparseResidualEncoder']


class SparseResidualEncoder(UResNetEncoder):
    """Encoder for sparse tensor feature extraction."""

    def __init__(self, coord_conv=False, pool_mode='avg',
                 feature_size=512, **cfg):
        """Initializes the sparse residual CNN encoder.

        Passes most of the configuration along to the underlying sparse
        residual CNN encoder defined in :class:`UResNetEncoder`.

        Parameters
        ----------
        coord_conv : bool, default False
            Whether to include normalized coordinates in the input features
        pool_mode : str, default 'avg'
            Final pooling operation mode ('avg', 'max' or 'conv'
        feature_size : int, default 512
            Number of features produced after the final pooling
        **cfg : dict, optional
            Configuration to pass along to the sparse residual encoder
        """
        # Initialize the parent class
        super().__init__(**cfg)

        # Store attributes
        self.coord_conv = coord_conv

        # Initialize the final pooling layer
        final_tensor_shape = self.spatial_size // (2**(self.depth-1))
        if self.pool_mode == 'avg':
            self.pool = ME.MinkowskiGlobalPooling()
        elif self.pool_mode == 'max':
            self.pool = torch.nn.Sequential(
                ME.MinkowskiMaxPooling(
                    final_tensor_shape, stride=final_tensor_shape,
                    dimension=self.dim),
                ME.MinkowskiGlobalPooling())
        elif self.pool_mode == 'conv':
            self.pool = torch.nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=self.num_planes[-1],
                    out_channels=self.num_planes[-1],
                    kernel_size=final_tensor_shape, stride=final_tensor_shape,
                    dimension=self.dim),
                ME.MinkowskiGlobalPooling())
        else:
            raise ValueError(
                    f"Pooling mode not recognized: {self.pool_mode}. Must be "
                     "one of 'avg', 'max' or 'conv'")

        # Initialize the final linear layer
        self.feature_size = feature_size
        self.linear = ME.MinkowskiLinear(
                self.num_planes[-1], self.feature_size)

    def forward(self, data):
        """Pass one batch of data through the CNN encoder.

        Parameters
        ----------
        data : torch.Tensor
             (N, 1 + D + N_f) Batch of data

        Returns
        -------
        torch.Tensor
            (B) Batch of features, one per batch ID
        """
        # Get the feature tensor
        features = data[:, VALUE_COL:VALUE_COL+self.num_input]
        if self.coord_conv:
            normalized_coords = data[:, COORD_COLS]/self.spatial_size
            features = torch.cat([normalized_coords, features], dim=1)

        # Build a sparse tensor
        x = ME.SparseTensor(coordinates=data[:, :VALUE_COL].int(),
                            features=features)

        # Pass through the CNN encoder
        output = self.encoder(x)
        final_tensor = output['final_tensor']

        # Pool the last layer
        z = self.pool(final_tensor)

        # Put it through a linear layer
        latent = self.linear(z)

        return latent.F
