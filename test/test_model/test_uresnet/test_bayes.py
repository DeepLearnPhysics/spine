"""Contracts for Bayesian UResNet segmentation and uncertainty outputs."""

import pytest

torch = pytest.importorskip("torch")

from spine.constants import VALUE_COL
from spine.data import TensorBatch
from spine.model.uresnet.bayes import BayesianSegmentationLoss, BayesianUResNet


def model_config(mode="standard", **kwargs):
    """Return a small Bayesian UResNet configuration for focused tests."""
    return {
        "num_input": 1,
        "num_classes": 3,
        "mode": mode,
        "num_samples": 2,
        "dropout_p": 0.25,
        "filters": 4,
        "depth": 2,
        "reps": 1,
        **kwargs,
    }


def input_batch():
    """Build two sparse events with one scalar feature per voxel."""
    data = torch.tensor(
        [
            [0, 0, 0, 0, 1.0],
            [0, 1, 0, 0, 2.0],
            [1, 0, 0, 0, 3.0],
            [1, 1, 0, 0, 4.0],
        ],
        dtype=torch.float32,
    )
    return TensorBatch(data, counts=torch.tensor([2, 2]))


def label_batch(values=(0, 1, 2, 1)):
    """Build semantic labels aligned with :func:`input_batch`."""
    labels = torch.zeros((len(values), VALUE_COL + 1), dtype=torch.float32)
    labels[:, VALUE_COL] = torch.tensor(values)
    return TensorBatch(labels, counts=torch.tensor([2, 2]))


@pytest.mark.model
@pytest.mark.parametrize("mode", ["standard", "mc_dropout", "evidential"])
def test_bayesian_uresnet_modes_return_aligned_batches(mode):
    """Each uncertainty mode must preserve voxel and batch alignment."""
    model = BayesianUResNet(model_config(mode))
    model.eval()

    result = model(input_batch())

    assert result["segmentation"].shape == (4, 3)
    assert result["segmentation"].counts.tolist() == [2, 2]
    if mode == "mc_dropout":
        assert result["softmax"].shape == (4, 3)
        assert torch.allclose(
            result["softmax"].torch_tensor().sum(dim=1),
            torch.ones(4),
        )
    elif mode == "evidential":
        evidence = result["evidence"].torch_tensor()
        concentration = result["concentration"].torch_tensor()
        assert torch.all(evidence >= 0)
        assert torch.allclose(concentration, evidence + 1.0)
        assert torch.allclose(
            result["expected_probability"].torch_tensor().sum(dim=1),
            torch.ones(4),
        )


def test_evidential_loss_is_finite_and_differentiable():
    """The evidential objective must consume evidence without shifting twice."""
    evidence = torch.rand((4, 3), requires_grad=True)
    predictions = TensorBatch(evidence, counts=torch.tensor([2, 2]))
    loss_fn = BayesianSegmentationLoss(
        model_config("evidential"),
        {"loss": "edl_sumsq", "annealing_steps": 10},
    )

    result = loss_fn(label_batch(), predictions, iteration=5)

    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert evidence.grad is not None


@pytest.mark.parametrize(
    ("mode", "loss"),
    [("standard", "edl_sumsq"), ("evidential", "ce")],
)
def test_loss_rejects_mode_mismatch(mode, loss):
    """Model output semantics and loss families cannot be mixed silently."""
    with pytest.raises(ValueError, match="requires"):
        BayesianSegmentationLoss(model_config(mode), {"loss": loss})


def test_bayesian_model_rejects_unknown_mode():
    """Unknown uncertainty behavior must fail during construction."""
    with pytest.raises(ValueError, match="Unknown Bayesian UResNet mode"):
        BayesianUResNet(model_config("mystery"))
