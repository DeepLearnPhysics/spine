"""Tests for reusable multi-objective loss balancing."""

import math

import pytest

from spine.model.common.loss_balancing import LossBalancer, LossTerm
from spine.utils.conditional import TORCH_AVAILABLE, torch

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")


def test_sum_balancing_is_parameter_free_and_preserves_scales():
    """The default policy should reproduce a conventional weighted sum."""
    balancer = LossBalancer({"classification": "categorical", "regression": "gaussian"})
    terms = {
        "classification": LossTerm(torch.tensor(2.0), "categorical", scale=3.0),
        "regression": LossTerm(torch.tensor(4.0), "gaussian", scale=0.5),
    }

    loss, diagnostics = balancer(terms)

    assert loss.item() == pytest.approx(8.0)
    assert not list(balancer.parameters())
    assert diagnostics["classification_weight"].item() == pytest.approx(3.0)
    assert diagnostics["regression_weight"].item() == pytest.approx(0.5)


def test_fixed_balancing_applies_user_priorities():
    """Explicit priorities should multiply producer-owned objective scales."""
    balancer = LossBalancer(
        {"first": "categorical", "second": "gaussian"},
        {"name": "fixed", "weights": {"first": 0.25, "second": 2.0}},
    )
    terms = {
        "first": LossTerm(torch.tensor(8.0), "categorical"),
        "second": LossTerm(torch.tensor(3.0), "gaussian", scale=2.0),
    }

    loss, diagnostics = balancer(terms)

    assert loss.item() == pytest.approx(14.0)
    assert diagnostics["first_weight"].item() == pytest.approx(0.25)
    assert diagnostics["second_weight"].item() == pytest.approx(4.0)


def test_uncertainty_balancing_uses_producer_families():
    """Likelihood families should select their code-owned Kendall formula."""
    balancer = LossBalancer(
        {"classification": "categorical", "regression": "gaussian"},
        {"name": "uncertainty"},
    )
    classification = torch.tensor(2.0, requires_grad=True)
    regression = torch.tensor(4.0, requires_grad=True)

    loss, diagnostics = balancer(
        {
            "classification": LossTerm(classification, "categorical"),
            "regression": LossTerm(regression, "gaussian"),
        }
    )
    loss.backward()

    # Both effective data coefficients start at one even though the Gaussian
    # likelihood contains an explicit one-half.
    assert loss.item() == pytest.approx(6.0)
    assert diagnostics["classification_weight"].item() == pytest.approx(1.0)
    assert diagnostics["regression_weight"].item() == pytest.approx(1.0)
    assert balancer.log_variances["classification"].grad.item() == pytest.approx(-1.5)
    assert balancer.log_variances["regression"].item() == pytest.approx(-math.log(2.0))
    assert balancer.log_variances["regression"].grad.item() == pytest.approx(-3.5)


def test_uncertainty_balancing_preserves_producer_priorities():
    """Native task coefficients should multiply the complete likelihood term."""
    balancer = LossBalancer(
        {"regression": "gaussian"},
        {"name": "uncertainty", "weights": {"regression": 3.0}},
    )

    loss, diagnostics = balancer(
        {"regression": LossTerm(torch.tensor(4.0), "gaussian", scale=2.0)}
    )

    assert loss.item() == pytest.approx(24.0)
    assert diagnostics["regression_weight"].item() == pytest.approx(6.0)


def test_uncertainty_balancing_supports_composite_stage_losses():
    """An aggregate stage should use a generic learned scale."""
    balancer = LossBalancer(
        {"stage": "composite"},
        {"name": "uncertainty"},
    )

    loss, diagnostics = balancer({"stage": LossTerm(torch.tensor(3.0), "composite")})
    loss.backward()

    assert loss.item() == pytest.approx(3.0)
    assert diagnostics["stage_weight"].item() == pytest.approx(1.0)
    assert balancer.log_variances["stage"].grad.item() == pytest.approx(-2.0)


def test_inactive_uncertainty_term_has_zero_parameter_gradient():
    """Missing supervision must suppress both data and regularizer updates."""
    balancer = LossBalancer(
        {"present": "categorical", "absent": "gaussian"},
        {"name": "uncertainty"},
    )
    loss, diagnostics = balancer(
        {
            "present": LossTerm(torch.tensor(1.0), "categorical"),
            "absent": LossTerm(torch.tensor(0.0), "gaussian", active=False),
        }
    )

    loss.backward()

    assert diagnostics["absent_active"].item() == 0.0
    assert diagnostics["absent_weight"].item() == 0.0
    assert balancer.log_variances["absent"].grad.item() == 0.0


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ([], TypeError, "configuration must be a mapping"),
        ({"name": "unknown"}, ValueError, "name"),
        ({"weights": {"missing": 1.0}}, ValueError, "Unknown"),
        ({"weights": {"task": -1.0}}, ValueError, "nonnegative"),
        ({"extra": True}, TypeError, "Unexpected"),
    ],
)
def test_loss_balancing_rejects_invalid_configuration(config, error, message):
    """Configuration errors should fail before the first training batch."""
    with pytest.raises(error, match=message):
        LossBalancer({"task": "categorical"}, config)
