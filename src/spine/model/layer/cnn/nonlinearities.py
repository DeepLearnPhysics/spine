"""Custom nonlinear activation functions for sparse tensors."""

from __future__ import annotations

import torch

from spine.model import sparse


class Mish(torch.nn.Module):
    r"""Apply the smooth, non-monotonic Mish activation to sparse features.

    Mish is defined element-wise as
    :math:`x \tanh(\operatorname{softplus}(x))`. Coordinates and sparse tensor
    provenance are preserved; only the feature matrix is replaced.

    References
    ----------
    .. [1] Misra, "Mish: A Self Regularized Non-Monotonic Activation
       Function," 2019. https://arxiv.org/abs/1908.08681
    """

    def forward(self, input_data: sparse.SparseTensor) -> sparse.SparseTensor:
        """Apply Mish to every sparse feature.

        Parameters
        ----------
        input_data : sparse.SparseTensor
            Sparse tensor whose feature matrix is transformed.

        Returns
        -------
        sparse.SparseTensor
            Tensor on the same coordinate map with Mish-transformed features.
        """
        out = input_data.features * torch.tanh(
            torch.nn.functional.softplus(input_data.features)
        )

        return input_data.replace_features(out)
