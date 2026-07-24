"""Backend-neutral functions that operate on sparse feature matrices."""

import torch

from .tensor import SparseTensor


def softmax(input: SparseTensor, dim: int = 1) -> SparseTensor:
    """Apply softmax to sparse features without changing coordinates.

    Parameters
    ----------
    input : SparseTensor
        Sparse tensor containing the features to normalize.
    dim : int, default 1
        Feature dimension along which to apply softmax.

    Returns
    -------
    SparseTensor
        Tensor on the same coordinate map with normalized features.
    """
    return input.replace_features(torch.softmax(input.F, dim=dim))
