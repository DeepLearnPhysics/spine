"""Tests for common dense projection and prediction layers."""

import numpy as np
import pytest
import torch

from spine.data import TensorBatch
from spine.model.common.evidential import EvidentialModel
from spine.model.common.final import FinalEvidential, FinalLinear, FinalMLP
from spine.model.common.mlp import MLP


def test_mlp_validates_widths_and_projects_features():
    """The shared MLP validates its topology and produces the final width."""
    model = MLP(
        3,
        depth=2,
        width=(5, 7),
        activation="relu",
        normalization="none",
    )

    assert model(torch.randn(4, 3)).shape == (4, 7)
    with pytest.raises(ValueError, match="once for each hidden layer"):
        MLP(
            3,
            depth=2,
            width=(5,),
            activation="relu",
            normalization="none",
        )


@pytest.mark.parametrize(
    ("head", "output_width"),
    [
        (FinalLinear(3, 2), 2),
        (
            FinalMLP(
                3,
                2,
                depth=1,
                width=4,
                activation="relu",
                normalization="none",
            ),
            2,
        ),
    ],
)
def test_final_heads_preserve_batching(head, output_width):
    """Dense prediction heads preserve TensorBatch entry counts."""
    features = TensorBatch(torch.randn(5, 3), counts=torch.tensor([2, 3]))

    output = head(features)

    assert output.torch_tensor().shape == (5, output_width)
    assert torch.equal(output.counts, features.counts)


def test_final_heads_require_torch_backing():
    """Dense prediction heads reject NumPy-backed batches explicitly."""
    features = TensorBatch(np.ones((2, 3), dtype=np.float32), counts=[2])

    with pytest.raises(TypeError, match="not backed by a torch"):
        FinalLinear(3, 2)(features)


def test_evidential_model_constructs_valid_distribution_parameters():
    """Evidential heads construct and constrain all four NIG parameters."""
    config = {
        "depth": 1,
        "width": 5,
        "activation": "relu",
        "normalization": "none",
    }
    model = EvidentialModel(3, config)
    output = model(torch.randn(4, 3))

    assert output.shape == (4, 4)
    assert torch.all(output[:, 1] >= 0)
    assert torch.all(output[:, 2] >= 1)
    assert torch.all(output[:, 3] >= 0)

    with pytest.raises(ValueError, match="exactly four"):
        FinalEvidential(3, 2, mlp=config)
