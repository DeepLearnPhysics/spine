"""Custom normalization layers."""

import MinkowskiEngine as ME
import torch


class MinkowskiPixelNorm(torch.nn.Module):
    """Pixel Normalization Layer for Sparse Tensors.

    PixelNorm layers were used in NVIDIA's ProGAN.

    This layer normalizes the feature vector in each
    pixel to unit length, and has no trainable weights.

    Original paper: https://arxiv.org/pdf/1710.10196.pdf
    """

    def __init__(self, eps=1e-8):
        """Initialize the normalization layer.

        Parameters
        ----------
        eps : float, default 1e-8
            Ensures non-divergent output features
        """
        # Initialize the parent class
        super(MinkowskiPixelNorm, self).__init__()

        # Store the parameters
        self.eps = eps

    def forward(self, input_data):
        """Pass tensor through the layer.

        Parameters
        ----------
        input_data : ME.SparseTensor
            Sparse input tensor

        Return
        ------
        ME.SparseTensor
            Sparse output tensor
        """
        features = input_data.F
        coords = input_data.C
        norm = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        out = features / (norm + self.eps).sqrt()

        return ME.SparseTensor(
            out,
            coordinate_manager=input_data.coordinate_manager,
            coordinate_map_key=input_data.coordinate_map_key,
        )

    def __repr__(self):
        """Representation of the noamlization layer.

        This includes the parameters of the layer.
        """
        suffix = f"({self.num_features}, eps={self.eps})"
        return self.__class__.__name__ + suffix


class MinkowskiAdaIN(torch.nn.Module):
    """Adaptive Instance Normalization Layer.

    Many parts of the code is borrowed from pytorch original
    `BatchNorm` implementation.

    Original paper: https://arxiv.org/pdf/1703.06868.pdf
    """

    def __init__(self, in_channels, eps=1e-5):
        """Initialize the normalization layer.

        Parameters
        ----------
        eps : float, default 1e-5
            Ensures non-divergent output features
        """
        # Initialize the parent class
        super(MinkowskiAdaIN, self).__init__()

        # Store parameters
        self.in_channels = in_channels
        self.eps = eps

        # Initialize weights and biases
        self._weight = torch.ones(in_channels)
        self._bias = torch.zeros(in_channels)

    @property
    def weight(self):
        """Weight parameter of the AdaIN layer.

        Note that in AdaptIS, the parameters to the AdaIN layer
        are trainable outputs from the controller network.
        """
        return self._weight

    @weight.setter
    def weight(self, weight):
        if weight.shape[0] != self.in_channels:
            raise ValueError(
                "Supplied weight vector feature dimension"
                "does not match AdaIN layer definition."
            )
        self._weight = weight

    @property
    def bias(self):
        """Bias parameter of the AdaIN layer.

        Note that in AdaptIS, the parameters to the AdaIN layer
        are trainable outputs from the controller network.
        """
        return self._bias

    @bias.setter
    def bias(self, bias):
        if bias.shape[0] != self.in_channels:
            raise ValueError(
                "Supplied bias vector feature dimension"
                "does not match AdaIN layer definition."
            )
        self._bias = bias

    def forward(self, x):
        """Pass tensor through the layer.

        Parameters
        ----------
        input_data : ME.SparseTensor
            Sparse input tensor

        Return
        ------
        ME.SparseTensor
            Sparse output tensor
        """
        f = x.F
        norm = (f - f.mean(dim=0)) / (f.var(dim=0) + self.eps).sqrt()
        out = self.weight * norm + self.bias

        return ME.SparseTensor(
            out, coords_key=input_data.coords_key, coords_manager=input_data.coords_man
        )
