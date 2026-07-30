"""Tests for common evidential prediction heads and objectives."""

import pytest
import torch

from spine.model.layer.common.evidential import (
    EDLRegressionLoss,
    EVDLoss,
    kld_evd_l2_loss,
    kld_regression_loss,
)


def test_evidential_classification_loss_reductions():
    """Classification loss produces per-example and reduced outputs."""
    alpha = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    labels = torch.tensor([0, 1])

    per_example = EVDLoss("edl_sumsq", num_classes=2, reduction="none")(alpha, labels)
    mean = EVDLoss("edl_sumsq", num_classes=2, reduction="mean")(alpha, labels)

    assert per_example.shape == (2,)
    assert mean.ndim == 0
    assert torch.allclose(mean, per_example.mean())


def test_evidential_regression_loss_honors_reduction():
    """Regression loss applies its configured reduction to both outputs."""
    logits = torch.tensor([[1.0, 1.0, 2.0, 1.0], [2.0, 1.0, 2.0, 1.0]])
    targets = torch.tensor([1.5, 2.5])

    total, nll = EDLRegressionLoss(reduction="mean")(logits, targets)

    assert total.ndim == 0
    assert nll.ndim == 0
    assert torch.isfinite(total)
    assert torch.isfinite(nll)


def test_evidential_losses_reject_invalid_configuration():
    """Invalid reduction and regularization names fail at construction."""
    with pytest.raises(ValueError, match="reduction"):
        EVDLoss("edl_sumsq", reduction="invalid")
    with pytest.raises(ValueError, match="KL Divergence"):
        EDLRegressionLoss(kl_mode="invalid")


@pytest.mark.parametrize(
    "regularizer",
    [kld_regression_loss, kld_evd_l2_loss],
)
def test_evidential_regularization_is_error_symmetric(regularizer):
    """Equal over- and under-prediction errors receive equal penalties."""
    logits = torch.tensor(
        [
            [0.5, 1.0, 2.0, 1.0],
            [1.5, 1.0, 2.0, 1.0],
        ]
    )
    targets = torch.tensor([1.0, 1.0])

    loss = regularizer(logits, targets)

    assert torch.allclose(loss[0], loss[1])
