"""Custom normalization layers for sparse tensors."""

from __future__ import annotations

import torch

from spine.model import sparse


class PixelNorm(torch.nn.Module):
    r"""Normalize each sparse site's feature vector to unit length [1]_.

    For a feature vector :math:`x_i`, this layer returns
    :math:`x_i / \sqrt{\sum_j x_{ij}^2 + \epsilon}`. It has no trainable
    parameters and does not alter coordinates.

    References
    ----------
    .. [1] Karras et al., "Progressive Growing of GANs for Improved Quality,
       Stability, and Variation," 2017. https://arxiv.org/abs/1710.10196
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the normalization layer.

        Parameters
        ----------
        eps : float, default 1e-8
            Positive numerical-stability term added to the squared norm.

        Raises
        ------
        ValueError
            If ``eps`` is not positive.
        """
        super().__init__()

        if eps <= 0.0:
            raise ValueError(f"`eps` must be positive, got {eps}.")
        self.eps = eps

    def forward(self, input_data: sparse.SparseTensor) -> sparse.SparseTensor:
        """Normalize each active site's feature vector.

        Parameters
        ----------
        input_data : sparse.SparseTensor
            Sparse tensor containing the feature vectors to normalize.

        Returns
        -------
        sparse.SparseTensor
            Tensor on the same coordinate map with normalized features.
        """
        features = input_data.features
        norm = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        out = features / (norm + self.eps).sqrt()

        return input_data.replace_features(out)

    def extra_repr(self) -> str:
        """Return the layer parameters included in ``repr``."""
        return f"eps={self.eps}"


class AdaIN(torch.nn.Module):
    """Apply adaptive instance normalization to sparse feature channels [1]_.

    The feature matrix is normalized independently by channel over all active
    sparse sites, then transformed with externally assignable scale and bias
    vectors. The affine vectors are buffers rather than trainable parameters
    because an external controller may replace them for each input.

    References
    ----------
    .. [1] Huang and Belongie, "Arbitrary Style Transfer in Real-time with
       Adaptive Instance Normalization," 2017.
       https://arxiv.org/abs/1703.06868
    """

    def __init__(self, in_channels: int, eps: float = 1e-5) -> None:
        """Initialize the normalization layer.

        Parameters
        ----------
        in_channels : int
            Number of channels in the sparse feature matrix.
        eps : float, default 1e-5
            Positive numerical-stability term added to the variance.

        Raises
        ------
        ValueError
            If ``in_channels`` or ``eps`` is not positive.
        """
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"`in_channels` must be positive, got {in_channels}.")
        if eps <= 0.0:
            raise ValueError(f"`eps` must be positive, got {eps}.")
        self.in_channels = in_channels
        self.eps = eps

        # These values may be replaced by a controller before each forward
        # pass, but registering their defaults keeps device moves and state
        # serialization correct.
        self.register_buffer("_weight", torch.ones(in_channels))
        self.register_buffer("_bias", torch.zeros(in_channels))

    @property
    def weight(self) -> torch.Tensor:
        """Return the channel-wise affine scale.

        Returns
        -------
        torch.Tensor
            Scale vector with shape ``(in_channels,)``.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: torch.Tensor) -> None:
        """Set the channel-wise affine scale.

        Parameters
        ----------
        weight : torch.Tensor
            Scale vector with shape ``(in_channels,)``.

        Raises
        ------
        ValueError
            If the scale has the wrong number of channels.
        """
        if weight.shape[0] != self.in_channels:
            raise ValueError(
                "Supplied weight vector feature dimension does not match "
                "the AdaIN layer definition."
            )
        self._weight = weight

    @property
    def bias(self) -> torch.Tensor:
        """Return the channel-wise affine bias.

        Returns
        -------
        torch.Tensor
            Bias vector with shape ``(in_channels,)``.
        """
        return self._bias

    @bias.setter
    def bias(self, bias: torch.Tensor) -> None:
        """Set the channel-wise affine bias.

        Parameters
        ----------
        bias : torch.Tensor
            Bias vector with shape ``(in_channels,)``.

        Raises
        ------
        ValueError
            If the bias has the wrong number of channels.
        """
        if bias.shape[0] != self.in_channels:
            raise ValueError(
                "Supplied bias vector feature dimension does not match "
                "the AdaIN layer definition."
            )
        self._bias = bias

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Normalize and affinely transform sparse features.

        Parameters
        ----------
        x : sparse.SparseTensor
            Sparse input whose feature width equals ``in_channels``.

        Returns
        -------
        sparse.SparseTensor
            Tensor on the same coordinate map with normalized features.
        """
        features = x.features
        normalized = (features - features.mean(dim=0)) / (
            features.var(dim=0, unbiased=False) + self.eps
        ).sqrt()
        out = self.weight * normalized + self.bias

        return x.replace_features(out)
