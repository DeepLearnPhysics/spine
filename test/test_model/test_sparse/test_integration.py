"""Integration tests for sparse model outputs and consumers."""

import torch

from spine.data import TensorBatch
from spine.io.unwrap import Unwrapper
from spine.model import sparse
from spine.model.manager import ModelManager
from spine.model.uresnet_ppn import UResNetPPN


def test_sparse_output_is_unwrapped_and_cast_to_numpy():
    """Unwrapping consumers accept sparse tensors and feature-map lists."""
    tensor = sparse.SparseTensor(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[0, 1, 2, 3], [2, 4, 5, 6]], dtype=torch.int32),
        batch_size=3,
    )

    unwrapper = Unwrapper()
    unwrapper.batch_size = 3
    entries = unwrapper._unwrap("feature_map", tensor)
    assert [len(entry) for entry in entries] == [1, 0, 1]

    levels = unwrapper._unwrap("decoder_tensors", [tensor, tensor])
    assert len(levels) == 3
    assert [len(level) for level in levels] == [2, 2, 2]

    result = {"decoder_tensors": [tensor]}
    ModelManager.cast_to_numpy(object.__new__(ModelManager), result)
    assert result["decoder_tensors"][0].is_numpy
    assert result["decoder_tensors"][0].counts.tolist() == [1, 0, 1]


def test_uresnet_accepts_an_empty_batch(uresnet):
    """A complete encoder-decoder remains defined with no active sites."""
    data = TensorBatch(torch.empty((0, 5)), counts=torch.tensor([0, 0]))

    result = uresnet(data)

    assert result["segmentation"].shape == (0, 3)
    assert result["segmentation"].counts.tolist() == [0, 0]
    assert result["final_tensor"].batch_size == 2
    assert result["decoder_tensors"][-1].shape == (0, 2)


def test_uresnet_restores_duplicate_input_rows(uresnet):
    """Point-aligned predictions retain the input multiplicity."""
    data = TensorBatch(
        torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 2.0],
                [0.0, 1.0, 1.0, 1.0, 4.0],
                [1.0, 3.0, 3.0, 3.0, 5.0],
            ]
        ),
        counts=torch.tensor([2, 1]),
    )

    result = uresnet(data)

    segmentation = result["segmentation"]
    assert segmentation.shape == (3, 3)
    assert segmentation.counts.tolist() == [2, 1]
    assert torch.equal(segmentation.tensor[0], segmentation.tensor[1])


def test_uresnet_ppn_accepts_an_empty_batch(uresnet_config):
    """The downstream PPN path consumes empty sparse feature maps."""
    data = TensorBatch(torch.empty((0, 5)), counts=torch.tensor([0, 0]))
    model = UResNetPPN(uresnet_config, {})

    result = model(data)

    assert result["segmentation"].shape == (0, 3)
    assert result["ppn_points"].counts.tolist() == [0, 0]
    assert result["ppn_points"].shape[0] == 0


def test_uresnet_ppn_separates_unique_and_restored_predictions(uresnet_config):
    """PPN retains unique predictions for loss and restored rows for users."""
    data = TensorBatch(
        torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 2.0],
                [0.0, 1.0, 1.0, 1.0, 4.0],
                [1.0, 3.0, 3.0, 3.0, 5.0],
            ]
        ),
        counts=torch.tensor([2, 1]),
    )
    model = UResNetPPN(uresnet_config, {})

    result = model(data)

    assert result["ppn_points"].counts.tolist() == [2, 1]
    assert result["ppn_points"].shape[0] == 3
    assert result["ppn_points_unique"].counts.tolist() == [1, 1]
    assert result["ppn_points_unique"].shape[0] == 2
    assert result["ppn_coords"][-1].shape[0] == 2
    assert torch.equal(
        result["ppn_points"].tensor[0],
        result["ppn_points"].tensor[1],
    )
