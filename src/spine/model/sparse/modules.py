"""Backend-selected, empty-safe sparse neural-network modules.

Each public class subclasses the corresponding native backend module so that
constructor signatures, parameters, and state-dictionary keys remain
compatible. The :class:`SparseTensor` wrapper is removed before a native
operation and restored afterward. Empty inputs bypass the backend entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
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
_NativeNetwork = backend.module("Network")

SpatialArg = int | Sequence[int]


class _NativeInitializer:
    """Initialize a dynamically selected backend class.

    Public wrappers expose canonical typed signatures. This method is the
    single dynamic boundary where those arguments are forwarded to the native
    backend implementation selected in :mod:`spine.model.sparse.backend`.
    """

    def _initialize_native(self, *args: Any, **kwargs: Any) -> None:
        native_init: Any = super().__init__
        native_init(*args, **kwargs)

    def _forward_native(self, *args: Any, **kwargs: Any) -> Any:
        """Forward through the dynamically selected backend implementation."""
        native_forward: Any = getattr(super(), "forward")
        return native_forward(*args, **kwargs)


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


class _EmptySafe(_NativeInitializer):
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
        if channels is None:
            linear = getattr(self, "linear", None)
            channels = getattr(linear, "out_features", None)
        return input.F.shape[1] if channels is None else int(channels)

    def forward(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply the native module or construct an empty result."""
        if not isinstance(input, SparseTensor):
            return self._forward_native(input, *args, **kwargs)
        if len(input) == 0:
            stride = input.tensor_stride
            if self._changes_stride:
                stride = _scaled_stride(input, self, self._transpose_stride)
            return SparseTensor.empty_like(input, self._empty_channels(input), stride)
        output = self._forward_native(input.backend_tensor, *args, **kwargs)
        return input._wrap(output)


class Convolution(_EmptySafe, _NativeConvolution):
    """Apply an empty-safe sparse convolution."""

    _changes_stride = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: SpatialArg = -1,
        stride: SpatialArg = 1,
        dilation: SpatialArg = 1,
        bias: bool = False,
        kernel_generator: Any | None = None,
        expand_coordinates: bool = False,
        convolution_mode: Any | None = None,
        dimension: int | None = None,
    ) -> None:
        """Initialize a sparse convolution using the canonical frontend API."""
        kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "dilation": dilation,
            "bias": bias,
            "kernel_generator": kernel_generator,
            "expand_coordinates": expand_coordinates,
            "dimension": dimension,
        }
        if convolution_mode is not None:
            kwargs["convolution_mode"] = convolution_mode
        self._initialize_native(in_channels, out_channels, **kwargs)


class ConvolutionTranspose(_EmptySafe, _NativeConvolutionTranspose):
    """Apply an empty-safe transposed sparse convolution."""

    _changes_stride = True
    _transpose_stride = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: SpatialArg = -1,
        stride: SpatialArg = 1,
        dilation: SpatialArg = 1,
        bias: bool = False,
        kernel_generator: Any | None = None,
        expand_coordinates: bool = False,
        convolution_mode: Any | None = None,
        dimension: int | None = None,
    ) -> None:
        """Initialize a transposed convolution using the frontend API."""
        kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
            "dilation": dilation,
            "bias": bias,
            "kernel_generator": kernel_generator,
            "expand_coordinates": expand_coordinates,
            "dimension": dimension,
        }
        if convolution_mode is not None:
            kwargs["convolution_mode"] = convolution_mode
        self._initialize_native(in_channels, out_channels, **kwargs)


class ChannelwiseConvolution(_EmptySafe, _NativeChannelwiseConvolution):
    """Apply an empty-safe channel-wise sparse convolution."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: SpatialArg = -1,
        stride: SpatialArg = 1,
        dilation: SpatialArg = 1,
        bias: bool = False,
        kernel_generator: Any | None = None,
        dimension: int = -1,
    ) -> None:
        """Initialize a channel-wise convolution using the frontend API."""
        self._initialize_native(
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            kernel_generator=kernel_generator,
            dimension=dimension,
        )


class Linear(_EmptySafe, _NativeLinear):
    """Apply an empty-safe linear transformation to sparse features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """Initialize a sparse linear transformation."""
        self._initialize_native(in_features, out_features, bias=bias)


class BatchNorm(_EmptySafe, _NativeBatchNorm):
    """Apply empty-safe batch normalization to sparse features."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """Initialize sparse batch normalization."""
        self._initialize_native(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


class InstanceNorm(_EmptySafe, _NativeInstanceNorm):
    """Apply empty-safe instance normalization to sparse features."""

    def __init__(self, num_features: int) -> None:
        """Initialize sparse instance normalization."""
        self._initialize_native(num_features)


class Dropout(_EmptySafe, _NativeDropout):
    """Apply dropout to sparse features, including empty tensors."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """Initialize sparse dropout."""
        self._initialize_native(p=p, inplace=inplace)


class ReLU(_EmptySafe, _NativeReLU):
    """Apply a rectified linear unit to sparse features."""

    def __init__(self, inplace: bool = False) -> None:
        """Initialize a sparse rectified linear unit."""
        self._initialize_native(inplace=inplace)


class PReLU(_EmptySafe, _NativePReLU):
    """Apply a parametric rectified linear unit to sparse features."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        """Initialize a sparse parametric rectified linear unit."""
        self._initialize_native(num_parameters=num_parameters, init=init)


class SELU(_EmptySafe, _NativeSELU):
    """Apply a scaled exponential linear unit to sparse features."""

    def __init__(self, inplace: bool = False) -> None:
        """Initialize a sparse SELU activation."""
        self._initialize_native(inplace=inplace)


class CELU(_EmptySafe, _NativeCELU):
    """Apply a continuously differentiable ELU to sparse features."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        """Initialize a sparse CELU activation."""
        self._initialize_native(alpha=alpha, inplace=inplace)


class LeakyReLU(_EmptySafe, _NativeLeakyReLU):
    """Apply a leaky rectified linear unit to sparse features."""

    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False,
    ) -> None:
        """Initialize a sparse leaky ReLU activation."""
        self._initialize_native(negative_slope=negative_slope, inplace=inplace)


class ELU(_EmptySafe, _NativeELU):
    """Apply an exponential linear unit to sparse features."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        """Initialize a sparse ELU activation."""
        self._initialize_native(alpha=alpha, inplace=inplace)


class Tanh(_EmptySafe, _NativeTanh):
    """Apply the hyperbolic tangent function to sparse features."""

    def __init__(self) -> None:
        """Initialize a sparse hyperbolic tangent activation."""
        self._initialize_native()


class Sigmoid(_EmptySafe, _NativeSigmoid):
    """Apply the logistic sigmoid function to sparse features."""

    def __init__(self) -> None:
        """Initialize a sparse sigmoid activation."""
        self._initialize_native()


class Softplus(_EmptySafe, _NativeSoftplus):
    """Apply the softplus function to sparse features."""

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        """Initialize a sparse softplus activation."""
        self._initialize_native(beta=beta, threshold=threshold)


class MaxPooling(_EmptySafe, _NativeMaxPooling):
    """Apply empty-safe max pooling and update the tensor stride."""

    _changes_stride = True

    def __init__(
        self,
        kernel_size: SpatialArg,
        stride: SpatialArg = 1,
        dilation: SpatialArg = 1,
        kernel_generator: Any | None = None,
        dimension: int | None = None,
    ) -> None:
        """Initialize sparse max pooling."""
        self._initialize_native(
            kernel_size,
            stride=stride,
            dilation=dilation,
            kernel_generator=kernel_generator,
            dimension=dimension,
        )


class AvgPooling(_EmptySafe, _NativeAvgPooling):
    """Apply empty-safe average pooling and update the tensor stride."""

    _changes_stride = True

    def __init__(
        self,
        kernel_size: SpatialArg = -1,
        stride: SpatialArg = 1,
        dilation: SpatialArg = 1,
        kernel_generator: Any | None = None,
        dimension: int | None = None,
    ) -> None:
        """Initialize sparse average pooling."""
        self._initialize_native(
            kernel_size,
            stride=stride,
            dilation=dilation,
            kernel_generator=kernel_generator,
            dimension=dimension,
        )


class SumPooling(_EmptySafe, _NativeSumPooling):
    """Apply empty-safe sum pooling and update the tensor stride."""

    _changes_stride = True

    def __init__(
        self,
        kernel_size: SpatialArg,
        stride: SpatialArg = 1,
        dilation: SpatialArg = 1,
        kernel_generator: Any | None = None,
        dimension: int | None = None,
    ) -> None:
        """Initialize sparse sum pooling."""
        self._initialize_native(
            kernel_size,
            stride=stride,
            dilation=dilation,
            kernel_generator=kernel_generator,
            dimension=dimension,
        )


class PoolingTranspose(_EmptySafe, _NativePoolingTranspose):
    """Apply empty-safe transposed pooling and reduce the tensor stride."""

    _changes_stride = True
    _transpose_stride = True

    def __init__(
        self,
        kernel_size: SpatialArg,
        stride: SpatialArg,
        dilation: SpatialArg = 1,
        kernel_generator: Any | None = None,
        expand_coordinates: bool = False,
        dimension: int | None = None,
    ) -> None:
        """Initialize sparse transposed pooling."""
        self._initialize_native(
            kernel_size,
            stride,
            dilation=dilation,
            kernel_generator=kernel_generator,
            expand_coordinates=expand_coordinates,
            dimension=dimension,
        )


class _GlobalPooling(_EmptySafe):
    """Mixin that defines global pooling for entirely empty batches.

    A global pool normally emits one row per batch entry. For an entirely
    empty input, this mixin returns one zero feature vector per entry so that
    downstream dense layers retain a well-defined batch dimension.
    """

    def __init__(self, mode: Any | None = None) -> None:
        """Initialize global pooling with an optional backend mode."""
        if mode is None:
            self._initialize_native()
        else:
            self._initialize_native(mode=mode)

    def forward(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply global pooling, synthesizing batch rows when necessary."""
        if isinstance(input, SparseTensor) and len(input) == 0 and input.batch_size:
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


class Pruning(_NativeInitializer, _NativePruning):
    """Remove sparse sites selected by a Boolean mask."""

    def __init__(self) -> None:
        """Initialize sparse pruning."""
        self._initialize_native()

    def forward(self, input: Any, mask: torch.Tensor) -> Any:
        """Prune active sites while preserving SPINE tensor metadata."""
        if not isinstance(input, SparseTensor):
            return super().forward(input, mask)
        if len(input) == 0:
            return SparseTensor.empty_like(input)
        return input._wrap(super().forward(input.backend_tensor, mask))


class Broadcast(_NativeInitializer, _NativeBroadcast):
    """Broadcast per-batch global features to every active sparse site."""

    def __init__(self) -> None:
        """Initialize sparse broadcasting."""
        self._initialize_native()

    def forward(self, input: Any, input_glob: Any) -> Any:
        """Broadcast global features onto a sparse tensor."""
        if not isinstance(input, SparseTensor):
            return super().forward(input, input_glob)
        if len(input) == 0:
            return SparseTensor.empty_like(input, input_glob.F.shape[1])
        return input._wrap(
            super().forward(input.backend_tensor, input_glob.backend_tensor)
        )


class BroadcastMultiplication(_NativeInitializer, _NativeBroadcastMultiplication):
    """Multiply active features by per-batch global features."""

    def __init__(self) -> None:
        """Initialize multiplicative sparse broadcasting."""
        self._initialize_native()

    def forward(self, input: Any, input_glob: Any) -> Any:
        """Broadcast and multiply global features onto a sparse tensor."""
        if not isinstance(input, SparseTensor):
            return super().forward(input, input_glob)
        if len(input) == 0:
            return SparseTensor.empty_like(input)
        return input._wrap(
            super().forward(input.backend_tensor, input_glob.backend_tensor)
        )


class Network(_NativeInitializer, _NativeNetwork):
    """Base class for sparse networks with a fixed spatial dimension."""

    def __init__(self, dimension: int) -> None:
        """Initialize a sparse network and expose its spatial dimension."""
        self.dimension = dimension
        self._initialize_native(dimension)


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
