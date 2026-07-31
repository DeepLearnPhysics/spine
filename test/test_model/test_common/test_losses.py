"""Tests for common custom loss modules."""

import pytest
import torch

from spine.model.common.losses import (
    BerHuLoss,
    BinaryFocalLoss,
    BinaryLogDiceCELoss,
    BinaryLogDiceCEMincutLoss,
    LogRMSE,
)


def test_custom_reduction_is_preserved_with_module_base():
    """Custom losses retain their documented reduction behavior."""
    loss = LogRMSE(reduction="mean")
    inputs = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.0, 4.0])

    result = loss(inputs, targets)

    assert result.ndim == 0
    assert loss.reduction == "mean"


def test_log_dice_ce_accepts_component_configuration():
    """Composite loss forwards mapping configurations to its components."""
    loss = BinaryLogDiceCELoss(
        log_dice={"eps": 1e-5},
        bce={"pos_weight": torch.tensor(2.0)},
    )

    assert loss.log_dice.eps == 1e-5
    assert torch.equal(loss.bce.pos_weight, torch.tensor(2.0))


def test_log_dice_ce_mincut_extends_log_dice_ce():
    """The min-cut composite initializes and evaluates all three components."""
    loss = BinaryLogDiceCEMincutLoss(w_mincut=0.25)
    logits = torch.tensor([-1.0, 1.0])
    targets = torch.tensor([0.0, 1.0])

    result = loss(logits, targets)

    assert isinstance(loss, BinaryLogDiceCELoss)
    assert result.ndim == 0
    assert torch.isfinite(result)


def test_binary_focal_loss_supports_probabilities_and_logits():
    """Focal loss honors its logits flag and documented reduction."""
    targets = torch.tensor([0.0, 1.0])
    probabilities = torch.tensor([0.25, 0.75])
    logits = torch.logit(probabilities)

    probability_loss = BinaryFocalLoss()(probabilities, targets)
    logits_loss = BinaryFocalLoss(logits=True)(logits, targets)

    assert probability_loss.shape == targets.shape
    assert torch.allclose(probability_loss, logits_loss)


def test_binary_focal_loss_applies_balancing_weights():
    """Class balancing changes per-sample focal-loss contributions."""
    inputs = torch.tensor([0.25, 0.25, 0.75])
    targets = torch.tensor([0.0, 0.0, 1.0])

    unweighted = BinaryFocalLoss()(inputs, targets)
    weighted = BinaryFocalLoss(balance_loss=True)(inputs, targets)

    assert not torch.allclose(unweighted, weighted)


def test_berhu_handles_perfect_predictions():
    """BerHu avoids division by zero when every residual is zero."""
    values = torch.ones(3)

    result = BerHuLoss(reduction="mean")(values, values)

    assert result == 0
    assert torch.isfinite(result)


@pytest.mark.parametrize(
    "loss_factory",
    [
        lambda: LogRMSE(reduction="invalid"),
        lambda: BerHuLoss(reduction="invalid"),
        lambda: BinaryFocalLoss(reduction="invalid"),
    ],
)
def test_losses_reject_invalid_reductions_at_construction(loss_factory):
    """Malformed loss reductions fail before the first training batch."""
    with pytest.raises(ValueError, match="Reduction"):
        loss_factory()
