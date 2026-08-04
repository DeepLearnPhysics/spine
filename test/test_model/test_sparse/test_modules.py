"""Tests for backend-selected, empty-safe sparse modules."""

import inspect

import pytest
import torch

from spine.model import sparse
from spine.model.sparse.modules import _scaled_stride, _stride_values

ME = pytest.importorskip("MinkowskiEngine")


@pytest.mark.parametrize(
    ("module", "parameters"),
    [
        (sparse.Network, ["dimension"]),
        (
            sparse.Convolution,
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "dilation",
                "bias",
                "kernel_generator",
                "expand_coordinates",
                "convolution_mode",
                "dimension",
            ],
        ),
        (
            sparse.ConvolutionTranspose,
            [
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "dilation",
                "bias",
                "kernel_generator",
                "expand_coordinates",
                "convolution_mode",
                "dimension",
            ],
        ),
        (
            sparse.ChannelwiseConvolution,
            [
                "in_channels",
                "kernel_size",
                "stride",
                "dilation",
                "bias",
                "kernel_generator",
                "dimension",
            ],
        ),
        (sparse.Linear, ["in_features", "out_features", "bias"]),
        (
            sparse.BatchNorm,
            [
                "num_features",
                "eps",
                "momentum",
                "affine",
                "track_running_stats",
            ],
        ),
        (sparse.InstanceNorm, ["num_features"]),
        (sparse.Dropout, ["p", "inplace"]),
        (sparse.ReLU, ["inplace"]),
        (sparse.PReLU, ["num_parameters", "init"]),
        (sparse.SELU, ["inplace"]),
        (sparse.CELU, ["alpha", "inplace"]),
        (sparse.LeakyReLU, ["negative_slope", "inplace"]),
        (sparse.ELU, ["alpha", "inplace"]),
        (sparse.Tanh, []),
        (sparse.Sigmoid, []),
        (sparse.Softplus, ["beta", "threshold"]),
        (
            sparse.MaxPooling,
            [
                "kernel_size",
                "stride",
                "dilation",
                "kernel_generator",
                "dimension",
            ],
        ),
        (
            sparse.AvgPooling,
            [
                "kernel_size",
                "stride",
                "dilation",
                "kernel_generator",
                "dimension",
            ],
        ),
        (
            sparse.SumPooling,
            [
                "kernel_size",
                "stride",
                "dilation",
                "kernel_generator",
                "dimension",
            ],
        ),
        (
            sparse.PoolingTranspose,
            [
                "kernel_size",
                "stride",
                "dilation",
                "kernel_generator",
                "expand_coordinates",
                "dimension",
            ],
        ),
        (sparse.GlobalPooling, ["mode"]),
        (sparse.GlobalAvgPooling, ["mode"]),
        (sparse.GlobalSumPooling, ["mode"]),
        (sparse.GlobalMaxPooling, ["mode"]),
        (sparse.Pruning, []),
        (sparse.Broadcast, []),
        (sparse.BroadcastMultiplication, []),
    ],
)
def test_public_constructor_contract(module, parameters):
    """Sparse frontend constructors expose stable ME-shaped signatures."""
    signature = inspect.signature(module)

    assert list(signature.parameters) == parameters
    assert all(
        parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )


def test_constructor_contract_rejects_stale_backend_arguments():
    """Dimension-free operations reject legacy backend-specific arguments."""
    with pytest.raises(TypeError):
        sparse.GlobalPooling(dimension=3)

    with pytest.raises(TypeError):
        sparse.Broadcast(dimension=3)


def test_optional_sparse_constructor_branches_initialize_at_runtime():
    """Optional backend modes and less common pointwise layers initialize."""
    convolution_mode = (
        inspect.signature(ME.MinkowskiConvolution)
        .parameters["convolution_mode"]
        .default
    )
    assert sparse.Convolution(
        1,
        1,
        kernel_size=1,
        convolution_mode=convolution_mode,
        dimension=2,
    )
    assert sparse.ConvolutionTranspose(
        1,
        1,
        kernel_size=1,
        convolution_mode=convolution_mode,
        dimension=2,
    )

    modules = (
        sparse.InstanceNorm(2),
        sparse.PReLU(2),
        sparse.SELU(),
        sparse.CELU(),
        sparse.ELU(),
        sparse.Tanh(),
    )
    assert all(module is not None for module in modules)
    pooling_mode = (
        inspect.signature(ME.MinkowskiGlobalPooling).parameters["mode"].default
    )
    assert sparse.GlobalPooling(mode=pooling_mode)


def test_network_exposes_spatial_dimension():
    """Sparse network subclasses inherit a canonical dimension attribute."""

    class ConcreteNetwork(sparse.Network):
        """Minimal concrete network used to exercise the base contract."""

        def forward(self, value):
            """Return an input unchanged."""
            return value

    network = ConcreteNetwork(dimension=3)

    assert network.dimension == 3


def test_empty_convolution_and_transpose_are_safe():
    """Empty tensors bypass backend convolution while preserving metadata."""
    tensor = sparse.SparseTensor(
        torch.empty((0, 2)),
        torch.empty((0, 4), dtype=torch.int32),
        batch_size=3,
    )
    conv = sparse.Convolution(2, 4, kernel_size=2, stride=2, dimension=3)
    deconv = sparse.ConvolutionTranspose(
        4,
        3,
        kernel_size=2,
        stride=2,
        dimension=3,
    )

    encoded = conv(tensor)
    decoded = deconv(encoded)

    assert encoded.shape == (0, 4)
    assert encoded.tensor_stride == (2, 2, 2)
    assert decoded.shape == (0, 3)
    assert decoded.tensor_stride == (1, 1, 1)
    assert decoded.counts.tolist() == [0, 0, 0]

    pooled = sparse.GlobalAvgPooling()(decoded)
    assert pooled.shape == (3, 3)
    assert pooled.counts.tolist() == [1, 1, 1]
    assert torch.count_nonzero(pooled.F) == 0


@pytest.mark.parametrize(
    "module",
    [
        sparse.Linear(2, 3),
        sparse.BatchNorm(2),
        sparse.Dropout(),
        sparse.ReLU(),
        sparse.Sigmoid(),
        sparse.Softplus(),
    ],
)
def test_pointwise_modules_accept_empty_tensors(module):
    """Point-wise operations preserve an empty sparse domain."""
    tensor = sparse.SparseTensor(
        torch.empty((0, 2)),
        torch.empty((0, 4), dtype=torch.int32),
        batch_size=2,
    )

    output = module(tensor)

    expected_channels = 3 if isinstance(module, sparse.Linear) else 2
    assert output.shape == (0, expected_channels)
    assert output.counts.tolist() == [0, 0]


def test_nonempty_modules_return_spine_sparse_tensors():
    """Native module results are wrapped back into the public tensor type."""
    tensor = sparse.SparseTensor(
        torch.tensor([[1.0, -2.0], [3.0, 4.0]]),
        torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32),
        batch_size=2,
    )

    output = sparse.ReLU()(sparse.Linear(2, 3)(tensor))

    assert isinstance(output, sparse.SparseTensor)
    assert output.shape == (2, 3)
    assert torch.equal(output.C, tensor.C)


def test_stride_helpers_accept_all_backend_representations():
    """Empty fallbacks normalize scalar, tensor, and generated strides."""
    tensor = sparse.SparseTensor(
        torch.empty((0, 1)),
        torch.empty((0, 3), dtype=torch.int32),
        tensor_stride=torch.tensor([4, 4]),
    )

    class KernelGenerator:
        kernel_stride = torch.tensor([2, 4])

    class GeneratedStride:
        kernel_generator = KernelGenerator()

    class UnitStride:
        pass

    assert _stride_values(3, 2) == (3, 3)
    assert _stride_values(torch.tensor([2, 3]), 2) == (2, 3)
    assert _scaled_stride(tensor, GeneratedStride()) == (8, 16)
    assert _scaled_stride(tensor, UnitStride(), transpose=True) == (4, 4)


def test_modules_continue_to_accept_native_tensors():
    """Native tensors pass directly through inherited backend forwards."""
    tensor = sparse.SparseTensor(
        torch.tensor([[-1.0], [2.0]]),
        torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32),
        batch_size=2,
    )

    output = sparse.ReLU()(tensor.backend_tensor)

    assert not isinstance(output, sparse.SparseTensor)
    assert torch.equal(output.F, torch.tensor([[0.0], [2.0]]))


def test_global_pooling_handles_nonempty_native_and_zero_batch_inputs():
    """Global pooling covers wrapped, native, and truly empty batch domains."""
    tensor = sparse.SparseTensor(
        torch.tensor([[1.0], [3.0], [2.0], [4.0]]),
        torch.tensor(
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 2, 0]],
            dtype=torch.int32,
        ),
        batch_size=2,
    )
    pool = sparse.GlobalAvgPooling()

    wrapped = pool(tensor)
    native = pool(tensor.backend_tensor)
    empty = sparse.SparseTensor(
        torch.empty((0, 1)),
        torch.empty((0, 3), dtype=torch.int32),
        batch_size=0,
    )

    assert isinstance(wrapped, sparse.SparseTensor)
    assert native.F.shape == (2, 1)
    assert pool(empty).shape == (0, 1)


def test_pruning_handles_wrapped_empty_and_native_inputs():
    """Pruning preserves wrappers and bypasses the backend for empties."""
    tensor = sparse.SparseTensor(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.int32),
    )
    empty = sparse.SparseTensor.empty_like(tensor)
    pruning = sparse.Pruning()

    wrapped = pruning(tensor, torch.tensor([True, False]))
    native = pruning(tensor.backend_tensor, torch.tensor([False, True]))

    assert isinstance(wrapped, sparse.SparseTensor)
    assert wrapped.F.tolist() == [[1.0]]
    assert native.F.tolist() == [[2.0]]
    assert pruning(empty, torch.empty(0, dtype=torch.bool)).shape == (0, 1)


def test_broadcast_operations_handle_wrapped_empty_and_native_inputs():
    """Broadcast wrappers cover normal, empty, and native backend domains."""
    tensor = sparse.SparseTensor(
        torch.tensor([[2.0], [4.0], [3.0], [5.0]]),
        torch.tensor(
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 2, 0]],
            dtype=torch.int32,
        ),
        batch_size=2,
    )
    global_tensor = sparse.GlobalAvgPooling()(tensor)
    empty = sparse.SparseTensor(
        torch.empty((0, 1)),
        torch.empty((0, 3), dtype=torch.int32),
        batch_size=2,
    )
    broadcast = sparse.Broadcast()
    multiply = sparse.BroadcastMultiplication()
    native_input = tensor.backend_tensor
    native_global = global_tensor.backend_tensor

    wrapped_broadcast = broadcast(tensor, global_tensor)
    native_broadcast = broadcast(native_input, native_global)
    wrapped_product = multiply(tensor, global_tensor)
    native_product = multiply(native_input, native_global)

    assert isinstance(wrapped_broadcast, sparse.SparseTensor)
    assert native_broadcast.F.shape == (4, 1)
    assert isinstance(wrapped_product, sparse.SparseTensor)
    assert native_product.F.tolist() == [[6.0], [12.0], [12.0], [20.0]]
    assert broadcast(empty, global_tensor).shape == (0, 1)
    assert multiply(empty, global_tensor).shape == (0, 1)
