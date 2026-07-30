"""Runtime tests for reusable sparse CNN blocks."""

import torch

from spine.model import sparse
from spine.model.layer.cnn.blocks import (
    ASPP,
    MBResConv,
    MBResConvSE,
    ResNetBlock,
    ResNeXtBlock,
    SEBlock,
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
