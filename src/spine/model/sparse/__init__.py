"""Backend-neutral sparse tensors and model operations.

The package is the public sparse API for SPINE models. It exposes semantic
operation names, an empty-safe :class:`SparseTensor`, and feature-wise
functions while keeping concrete sparse-convolution engines behind an adapter.
"""

from .functional import softmax
from .modules import (
    CELU,
    ELU,
    SELU,
    AvgPooling,
    BatchNorm,
    Broadcast,
    BroadcastMultiplication,
    ChannelwiseConvolution,
    Convolution,
    ConvolutionTranspose,
    Dropout,
    GlobalAvgPooling,
    GlobalMaxPooling,
    GlobalPooling,
    GlobalSumPooling,
    InstanceNorm,
    LeakyReLU,
    Linear,
    MaxPooling,
    Network,
    PoolingTranspose,
    PReLU,
    Pruning,
    ReLU,
    Sigmoid,
    Softplus,
    SumPooling,
    Tanh,
)
from .tensor import SparseTensor


def cat(*inputs):
    """Concatenate sparse feature matrices on a shared coordinate map.

    Parameters
    ----------
    *inputs : SparseTensor or sequence of SparseTensor
        Sparse tensors with identical coordinates. A single list or tuple is
        accepted for compatibility with the native backend API.

    Returns
    -------
    SparseTensor
        Sparse tensor whose feature dimension is the sum of the input feature
        dimensions. Input provenance is inherited from the first tensor.

    Raises
    ------
    ValueError
        If no input tensors are provided.

    Notes
    -----
    If every input is empty, concatenation is performed without entering the
    backend and preserves the empty tensor's batch and stride metadata.
    """
    from . import backend

    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        inputs = tuple(inputs[0])
    if not inputs:
        raise ValueError("Sparse concatenation requires at least one tensor.")
    if not all(isinstance(value, SparseTensor) for value in inputs):
        return backend.concatenate(*inputs)
    if not any(len(value) for value in inputs):
        channels = sum(value.F.shape[1] for value in inputs)
        return SparseTensor.empty_like(inputs[0], channels)
    output = backend.concatenate(*(value.backend_tensor for value in inputs))
    return inputs[0]._wrap(output)


__all__ = [
    "SparseTensor",
    "Network",
    "Convolution",
    "ConvolutionTranspose",
    "ChannelwiseConvolution",
    "Linear",
    "BatchNorm",
    "InstanceNorm",
    "Dropout",
    "ReLU",
    "PReLU",
    "SELU",
    "CELU",
    "LeakyReLU",
    "ELU",
    "Tanh",
    "Sigmoid",
    "Softplus",
    "MaxPooling",
    "AvgPooling",
    "SumPooling",
    "PoolingTranspose",
    "GlobalPooling",
    "GlobalAvgPooling",
    "GlobalSumPooling",
    "GlobalMaxPooling",
    "Pruning",
    "Broadcast",
    "BroadcastMultiplication",
    "cat",
    "softmax",
]
