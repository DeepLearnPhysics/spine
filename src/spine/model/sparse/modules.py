"""Backend-selected, empty-safe sparse neural-network modules.

Each public class subclasses the corresponding native backend module so that
constructor signatures, parameters, and state-dictionary keys remain
compatible. The :class:`SparseTensor` wrapper is removed before a native
operation and restored afterward. Empty inputs bypass the backend entirely.
"""

from __future__ import annotations

from typing import Any

import torch

from . import backend
from .tensor import SparseTensor

_NativeConvolution = backend.module("Convolution")
_NativeConvolutionTranspose = backend.module("ConvolutionTranspose")
_NativeChannelwiseConvolution = backend.module("ChannelwiseConvolution")
_NativeLinear = backend.module("Linear")
_NativeBatchNorm = backend.module("BatchNorm")
_NativeInstanceNorm = backend.module("InstanceNorm")
_NativeDropout = backend.module("Dropout")
_NativeReLU = backend.module("ReLU")
_NativePReLU = backend.module("PReLU")
_NativeSELU = backend.module("SELU")
_NativeCELU = backend.module("CELU")
_NativeLeakyReLU = backend.module("LeakyReLU")
_NativeELU = backend.module("ELU")
_NativeTanh = backend.module("Tanh")
_NativeSigmoid = backend.module("Sigmoid")
_NativeSoftplus = backend.module("Softplus")
_NativeMaxPooling = backend.module("MaxPooling")
_NativeAvgPooling = backend.module("AvgPooling")
_NativeSumPooling = backend.module("SumPooling")
_NativePoolingTranspose = backend.module("PoolingTranspose")
_NativeGlobalPooling = backend.module("GlobalPooling")
_NativeGlobalAvgPooling = backend.module("GlobalAvgPooling")
_NativeGlobalSumPooling = backend.module("GlobalSumPooling")
_NativeGlobalMaxPooling = backend.module("GlobalMaxPooling")
_NativePruning = backend.module("Pruning")
_NativeBroadcast = backend.module("Broadcast")
_NativeBroadcastMultiplication = backend.module("BroadcastMultiplication")


def _stride_values(value: Any, dimension: int) -> tuple[int, ...]:
    """Normalize a scalar or vector stride to one value per dimension."""
    if isinstance(value, int):
        return (value,) * dimension
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return tuple(int(v) for v in value)


def _scaled_stride(
    tensor: SparseTensor, layer: Any, transpose: bool = False
) -> tuple[int, ...]:
    """Compute the output stride of an empty convolution or pooling layer."""
    stride = getattr(layer, "stride", None)
    if stride is None and hasattr(layer, "kernel_generator"):
        stride = layer.kernel_generator.kernel_stride
    factor = _stride_values(1 if stride is None else stride, tensor.dimension)
    if transpose:
        return tuple(max(1, a // b) for a, b in zip(tensor.tensor_stride, factor))
    return tuple(a * b for a, b in zip(tensor.tensor_stride, factor))


class _EmptySafe:
    """Mixin that unwraps SPINE tensors and wraps backend results.

    Native inputs continue through the underlying backend unchanged. Empty
    SPINE tensors instead produce metadata-preserving empty outputs, avoiding
    backend kernels that require at least one active site.
    """

    out_channels: int | None = None
    _changes_stride = False
    _transpose_stride = False

    def _empty_channels(self, input: SparseTensor) -> int:
        """Infer the output feature count for an empty input."""
        channels = getattr(self, "out_channels", None)
        if channels is None:
            channels = getattr(self, "out_features", None)
        if channels is None and hasattr(self, "linear"):
            channels = getattr(self.linear, "out_features", None)
        return input.F.shape[1] if channels is None else int(channels)

    def forward(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply the native module or construct an empty result."""
        if not isinstance(input, SparseTensor):
            return super().forward(input, *args, **kwargs)
        if not len(input):
            stride = input.tensor_stride
            if self._changes_stride:
                stride = _scaled_stride(input, self, self._transpose_stride)
            return SparseTensor.empty_like(input, self._empty_channels(input), stride)
        output = super().forward(input.backend_tensor, *args, **kwargs)
        return input._wrap(output)


class Convolution(_EmptySafe, _NativeConvolution):
    """Apply an empty-safe sparse convolution."""

    _changes_stride = True


class ConvolutionTranspose(_EmptySafe, _NativeConvolutionTranspose):
    """Apply an empty-safe transposed sparse convolution."""

    _changes_stride = True
    _transpose_stride = True


class ChannelwiseConvolution(_EmptySafe, _NativeChannelwiseConvolution):
    """Apply an empty-safe channel-wise sparse convolution."""


class Linear(_EmptySafe, _NativeLinear):
    """Apply an empty-safe linear transformation to sparse features."""


class BatchNorm(_EmptySafe, _NativeBatchNorm):
    """Apply empty-safe batch normalization to sparse features."""


class InstanceNorm(_EmptySafe, _NativeInstanceNorm):
    """Apply empty-safe instance normalization to sparse features."""


class Dropout(_EmptySafe, _NativeDropout):
    """Apply dropout to sparse features, including empty tensors."""


class ReLU(_EmptySafe, _NativeReLU):
    """Apply a rectified linear unit to sparse features."""


class PReLU(_EmptySafe, _NativePReLU):
    """Apply a parametric rectified linear unit to sparse features."""


class SELU(_EmptySafe, _NativeSELU):
    """Apply a scaled exponential linear unit to sparse features."""


class CELU(_EmptySafe, _NativeCELU):
    """Apply a continuously differentiable ELU to sparse features."""


class LeakyReLU(_EmptySafe, _NativeLeakyReLU):
    """Apply a leaky rectified linear unit to sparse features."""


class ELU(_EmptySafe, _NativeELU):
    """Apply an exponential linear unit to sparse features."""


class Tanh(_EmptySafe, _NativeTanh):
    """Apply the hyperbolic tangent function to sparse features."""


class Sigmoid(_EmptySafe, _NativeSigmoid):
    """Apply the logistic sigmoid function to sparse features."""


class Softplus(_EmptySafe, _NativeSoftplus):
    """Apply the softplus function to sparse features."""


class MaxPooling(_EmptySafe, _NativeMaxPooling):
    """Apply empty-safe max pooling and update the tensor stride."""

    _changes_stride = True


class AvgPooling(_EmptySafe, _NativeAvgPooling):
    """Apply empty-safe average pooling and update the tensor stride."""

    _changes_stride = True


class SumPooling(_EmptySafe, _NativeSumPooling):
    """Apply empty-safe sum pooling and update the tensor stride."""

    _changes_stride = True


class PoolingTranspose(_EmptySafe, _NativePoolingTranspose):
    """Apply empty-safe transposed pooling and reduce the tensor stride."""

    _changes_stride = True
    _transpose_stride = True


class _GlobalPooling(_EmptySafe):
    """Mixin that defines global pooling for entirely empty batches.

    A global pool normally emits one row per batch entry. For an entirely
    empty input, this mixin returns one zero feature vector per entry so that
    downstream dense layers retain a well-defined batch dimension.
    """

    def forward(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply global pooling, synthesizing batch rows when necessary."""
        if isinstance(input, SparseTensor) and not len(input) and input.batch_size:
            coordinates = input.C.new_zeros((input.batch_size, input.dimension + 1))
            coordinates[:, 0] = torch.arange(
                input.batch_size, device=input.C.device, dtype=input.C.dtype
            )
            features = input.F.new_zeros((input.batch_size, input.F.shape[1]))
            return SparseTensor(
                features,
                coordinates,
                tensor_stride=input.tensor_stride,
                batch_size=input.batch_size,
            )
        return super().forward(input, *args, **kwargs)


class GlobalPooling(_GlobalPooling, _NativeGlobalPooling):
    """Apply the backend's general global pooling operation."""


class GlobalAvgPooling(_GlobalPooling, _NativeGlobalAvgPooling):
    """Average sparse features independently for each batch entry."""


class GlobalSumPooling(_GlobalPooling, _NativeGlobalSumPooling):
    """Sum sparse features independently for each batch entry."""


class GlobalMaxPooling(_GlobalPooling, _NativeGlobalMaxPooling):
    """Take the feature-wise maximum for each batch entry."""


class Pruning(_NativePruning):
    """Remove sparse sites selected by a Boolean mask."""

    def forward(self, input: Any, mask: torch.Tensor) -> Any:
        """Prune active sites while preserving SPINE tensor metadata."""
        if not isinstance(input, SparseTensor):
            return super().forward(input, mask)
        if not len(input):
            return SparseTensor.empty_like(input)
        return input._wrap(super().forward(input.backend_tensor, mask))


class Broadcast(_NativeBroadcast):
    """Broadcast per-batch global features to every active sparse site."""

    def forward(self, input: Any, input_glob: Any) -> Any:
        """Broadcast global features onto a sparse tensor."""
        if not isinstance(input, SparseTensor):
            return super().forward(input, input_glob)
        if not len(input):
            return SparseTensor.empty_like(input, input_glob.F.shape[1])
        return input._wrap(
            super().forward(input.backend_tensor, input_glob.backend_tensor)
        )


class BroadcastMultiplication(_NativeBroadcastMultiplication):
    """Multiply active features by per-batch global features."""

    def forward(self, input: Any, input_glob: Any) -> Any:
        """Broadcast and multiply global features onto a sparse tensor."""
        if not isinstance(input, SparseTensor):
            return super().forward(input, input_glob)
        if not len(input):
            return SparseTensor.empty_like(input)
        return input._wrap(
            super().forward(input.backend_tensor, input_glob.backend_tensor)
        )


Network = backend.module("Network")

__all__ = [
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
]
