"""Implement the SPINE sparse backend contract with MinkowskiEngine.

This is the only module in :mod:`spine.model` that imports MinkowskiEngine.
Keeping native names and tensor access here prevents backend details from
leaking into model definitions.
"""

from __future__ import annotations

from typing import Any

try:
    import MinkowskiEngine as engine
except ModuleNotFoundError as exc:
    raise ImportError(
        "MinkowskiEngine is required for the `minkowski` sparse backend."
    ) from exc


_MODULES = {
    "Network": engine.MinkowskiNetwork,
    "Convolution": engine.MinkowskiConvolution,
    "ConvolutionTranspose": engine.MinkowskiConvolutionTranspose,
    "ChannelwiseConvolution": engine.MinkowskiChannelwiseConvolution,
    "Linear": engine.MinkowskiLinear,
    "BatchNorm": engine.MinkowskiBatchNorm,
    "InstanceNorm": engine.MinkowskiInstanceNorm,
    "Dropout": engine.MinkowskiDropout,
    "ReLU": engine.MinkowskiReLU,
    "PReLU": engine.MinkowskiPReLU,
    "SELU": engine.MinkowskiSELU,
    "CELU": engine.MinkowskiCELU,
    "LeakyReLU": engine.MinkowskiLeakyReLU,
    "ELU": engine.MinkowskiELU,
    "Tanh": engine.MinkowskiTanh,
    "Sigmoid": engine.MinkowskiSigmoid,
    "Softplus": engine.MinkowskiSoftplus,
    "MaxPooling": engine.MinkowskiMaxPooling,
    "AvgPooling": engine.MinkowskiAvgPooling,
    "SumPooling": engine.MinkowskiSumPooling,
    "PoolingTranspose": engine.MinkowskiPoolingTranspose,
    "GlobalPooling": engine.MinkowskiGlobalPooling,
    "GlobalAvgPooling": engine.MinkowskiGlobalAvgPooling,
    "GlobalSumPooling": engine.MinkowskiGlobalSumPooling,
    "GlobalMaxPooling": engine.MinkowskiGlobalMaxPooling,
    "Pruning": engine.MinkowskiPruning,
    "Broadcast": engine.MinkowskiBroadcast,
    "BroadcastMultiplication": engine.MinkowskiBroadcastMultiplication,
}


def module(operation: str) -> type:
    """Resolve a semantic SPINE operation to its native module class.

    Parameters
    ----------
    operation : str
        Backend-neutral operation name.

    Returns
    -------
    type
        MinkowskiEngine implementation of the operation.

    Raises
    ------
    ValueError
        If the operation is not implemented by this adapter.
    """
    try:
        return _MODULES[operation]
    except KeyError as exc:
        raise ValueError(
            f"The Minkowski backend does not implement `{operation}`."
        ) from exc


def create_tensor(**kwargs: Any) -> Any:
    """Create a native MinkowskiEngine sparse tensor."""
    return engine.SparseTensor(**kwargs)


def concatenate(*tensors: Any) -> Any:
    """Concatenate native tensors with a common coordinate map."""
    return engine.cat(*tensors)


def coordinates(tensor: Any) -> Any:
    """Return a native tensor's batched coordinate matrix."""
    return tensor.C


def features(tensor: Any) -> Any:
    """Return a native tensor's feature matrix."""
    return tensor.F


def tensor_stride(tensor: Any) -> tuple[int, ...]:
    """Return a native tensor's spatial stride as a tuple."""
    return tuple(int(value) for value in tensor.tensor_stride)


def coordinate_map_key(tensor: Any) -> Any:
    """Return a native tensor's coordinate-map key."""
    return tensor.coordinate_map_key


def coordinate_manager(tensor: Any) -> Any:
    """Return a native tensor's coordinate manager."""
    return tensor.coordinate_manager


def features_at_coordinates(tensor: Any, queries: Any) -> Any:
    """Query native tensor features at continuous coordinates."""
    return tensor.features_at_coordinates(queries)
