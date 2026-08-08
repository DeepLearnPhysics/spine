"""Runtime tests for reusable sparse CNN blocks."""

import pytest
import torch

from spine.model import sparse
from spine.model.cnn.blocks import (
    ASPP,
    SPP,
    AtrousIIBlock,
    CascadeDilationBlock,
    ConvolutionBlock,
    DropoutBlock,
    MBConv,
    MBResConv,
    MBResConvSE,
    ResNetBlock,
    ResNeXtBlock,
    SEBlock,
    SEResNetBlock,
)


def make_sparse(table):
    return sparse.SparseTensor(
        coordinates=table[:, :4].int(),
        features=table[:, 4:],
        batch_size=2,
    )


def test_resnet_stride_downsamples_residual_and_main_path(sparse_table):
    tensor = make_sparse(sparse_table)
    block = ResNetBlock(
        1,
        4,
        stride=2,
        normalization="none",
        dimension=3,
    )

    output = block(tensor)

    assert output.features.shape[1] == 4
    assert output.tensor_stride == (2, 2, 2)


def test_se_block_applies_its_attention(sparse_table):
    tensor = make_sparse(sparse_table)
    tensor = tensor.replace_features(tensor.features.repeat(1, 4))

    output = SEBlock(4, ratio=2)(tensor)

    assert output.features.shape == tensor.features.shape
    assert not torch.equal(output.features, tensor.features)


def test_mb_residual_se_returns_one_residual_application(sparse_table):
    tensor = make_sparse(sparse_table)
    tensor = tensor.replace_features(tensor.features.repeat(1, 4))
    block = MBResConvSE(
        4,
        4,
        se_ratio=2,
        normalization="none",
        dimension=3,
    )

    output = block(tensor)

    assert output.features.shape == tensor.features.shape


def test_mb_residual_se_stride_downsamples_residual(sparse_table):
    """SE mobile residuals project their skip path when downsampling."""
    output = MBResConvSE(
        1,
        4,
        stride=2,
        se_ratio=2,
        normalization="none",
    )(make_sparse(sparse_table))
    assert output.tensor_stride == (2, 2, 2)


def test_mb_residual_stride_downsamples_once(sparse_table):
    tensor = make_sparse(sparse_table)
    block = MBResConv(
        1,
        4,
        stride=2,
        normalization="none",
        dimension=3,
    )

    output = block(tensor)

    assert output.tensor_stride == (2, 2, 2)
    assert output.features.shape[1] == 4


def test_resnext_stride_keeps_paths_and_residual_aligned(sparse_table):
    tensor = make_sparse(sparse_table)
    tensor = tensor.replace_features(tensor.features.repeat(1, 4))
    block = ResNeXtBlock(
        4,
        4,
        cardinality=2,
        strides=2,
        dilations=(1, 2),
        normalization="none",
        dimension=3,
    )

    output = block(tensor)

    assert output.tensor_stride == (2, 2, 2)


def test_aspp_supports_different_input_and_output_widths(sparse_table):
    tensor = make_sparse(sparse_table)
    block = ASPP(1, 2, width=2, dilations=(1, 2), dimension=3)

    output = block(tensor)

    assert output.features.shape[1] == 2


@pytest.mark.parametrize(
    "block",
    [
        ConvolutionBlock(1, 2, normalization="none"),
        DropoutBlock(1, 2, p=0.2, normalization="none"),
        AtrousIIBlock(1, 2, normalization="none"),
        CascadeDilationBlock(1, 2, depth=2, dilations=(1, 2)),
        MBConv(1, 2, expand_ratio=1, normalization="none"),
        MBConv(1, 2, expand_ratio=2, normalization="none"),
        SEResNetBlock(1, 2, se_ratio=2, normalization="none"),
    ],
)
def test_sparse_blocks_preserve_rows_and_set_output_width(block, sparse_table):
    """Maintained sparse blocks return the configured feature width."""
    output = block(make_sparse(sparse_table))

    assert output.features.shape == (len(sparse_table), 2)


@pytest.mark.parametrize(
    "block",
    [
        ResNetBlock(2, 2, normalization="none"),
        AtrousIIBlock(2, 2, normalization="none"),
        ResNeXtBlock(
            2,
            2,
            cardinality=2,
            dilations=1,
            kernel_sizes=(3, 3),
            normalization="none",
        ),
        ASPP(2, 2, width=5, dilations=None),
        CascadeDilationBlock(2, 2, depth=6, dilations=None),
        SPP(2, 2, kernel_sizes=(2,), dilations=None),
    ],
)
def test_sparse_blocks_cover_default_and_identity_paths(block, sparse_table):
    """Default dilation and same-width residual paths remain executable."""
    tensor = make_sparse(sparse_table).replace_features(
        sparse_table[:, 4:].repeat(1, 2)
    )
    assert block(tensor).features.shape[1] == 2


@pytest.mark.parametrize("mode", ["avg", "max", "sum"])
def test_spatial_pyramid_pooling_supports_all_pool_modes(mode, sparse_table):
    """Global and local pyramid branches produce row-aligned features."""
    block = SPP(1, 2, kernel_sizes=(2,), dilations=1, mode=mode)

    output = block(make_sparse(sparse_table))

    assert output.features.shape == (len(sparse_table), 2)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ConvolutionBlock(1, 1, dimension=0), "dimension > 0"),
        (lambda: ResNetBlock(1, 1, dimension=0), "dimension > 0"),
        (lambda: DropoutBlock(1, 1, dimension=0), "dimension > 0"),
        (lambda: DropoutBlock(1, 1, p=1.0), "must be in"),
        (lambda: AtrousIIBlock(1, 1, dimension=0), "dimension > 0"),
        (lambda: MBConv(1, 1, expand_ratio=0), "must be positive"),
        (lambda: SEBlock(2, ratio=0), "must be positive"),
        (lambda: SEResNetBlock(1, 1, dimension=0), "dimension > 0"),
        (lambda: CascadeDilationBlock(1, 1, depth=2, dilations=(1,)), "depth"),
        (lambda: ASPP(1, 1, width=2, dilations=(1,)), "width"),
        (lambda: SPP(1, 1, mode="bad"), "Invalid pooling mode"),
        (
            lambda: SPP(1, 1, kernel_sizes=(2, 4), dilations=(1,)),
            "len\\(kernel_sizes\\)",
        ),
        (lambda: SPP(1, 1, kernel_sizes=(2,), dilations=object()), "dilations"),
    ],
)
def test_sparse_blocks_reject_invalid_hyperparameters(factory, message):
    """Reusable blocks reject malformed structural hyperparameters."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dimension": 0}, "dimension > 0"),
        ({"cardinality": 0}, "cardinality > 0"),
        ({"in_features": 3}, "divisible"),
        ({"dilations": (1,)}, "len\\(dilations\\)"),
        ({"kernel_sizes": (3,)}, "len\\(kernel_sizes\\)"),
        ({"strides": (1,)}, "len\\(strides\\)"),
        ({"dilations": object()}, "Invalid type"),
        ({"kernel_sizes": object()}, "Invalid type"),
        ({"strides": object()}, "Invalid type"),
        ({"strides": (1, 2)}, "same stride"),
    ],
)
def test_resnext_validates_path_configuration(kwargs, message):
    """Every ResNeXt path must have compatible dimensions and resolution."""
    config = {
        "in_features": 4,
        "out_features": 4,
        "cardinality": 2,
        "normalization": "none",
    }
    config.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ResNeXtBlock(**config)
