"""Tests for common evidential prediction heads and objectives."""

import pytest
import torch

from spine.model.common.evidential import (
    EDLRegressionLoss,
    EVDLoss,
    EvidentialModel,
    _reduce_loss,
    evd_kl_divergence,
    evd_loss_factory,
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


def test_evidential_model_validates_epsilon_and_supports_logspace():
    """The prediction head constrains NIG parameters in either mean domain."""
    with pytest.raises(ValueError, match="nonnegative"):
        EvidentialModel(2, {"depth": 1, "width": 3}, eps=-1.0)

    model = EvidentialModel(
        2,
        {
            "depth": 1,
            "width": 3,
            "activation": "relu",
            "normalization": "none",
        },
        eps=0.1,
        logspace=True,
    )
    output = model(torch.randn(4, 2))
    assert output.shape == (4, 4)
    assert torch.all((output[:, 0] >= 0) & (output[:, 0] <= 2))
    assert torch.all(output[:, 1:] >= 0.1)


@pytest.mark.parametrize("name", ["edl_digamma", "edl_sumsq", "edl_nll"])
def test_all_evidential_classification_risks_run(name):
    """Every registered Dirichlet Bayes-risk formulation is executable."""
    alpha = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    labels = torch.tensor([0, 1])
    loss = EVDLoss(name, num_classes=2, reduction="sum")(alpha, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_evidential_classification_supports_evidence_and_soft_targets():
    """Evidence inputs and pre-expanded targets follow the same objective."""
    evidence = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss_fn = EVDLoss(
        "edl_nll",
        num_classes=2,
        one_hot=False,
        mode="evidence",
        annealing_steps=2,
    )
    loss = loss_fn(evidence, targets, iteration=2)
    assert loss.shape == (2,)
    assert "mode=evidence" in str(loss_fn)

    beta = torch.full_like(evidence, 2.0)
    assert torch.isfinite(evd_kl_divergence(evidence + 1.0, beta)).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"annealing_steps": 0}, "must be positive"),
        ({"num_classes": 1}, "requires two classes"),
        ({"mode": "bad"}, "input mode"),
    ],
)
def test_evidential_classification_validates_configuration(kwargs, message):
    """Invalid annealing, class count, and input modes fail at construction."""
    with pytest.raises(ValueError, match=message):
        EVDLoss("edl_sumsq", **kwargs)
    with pytest.raises(ValueError, match="Unknown evidential loss"):
        evd_loss_factory("bad")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reduction": "bad"}, "reduction"),
        ({"regularization_weight": -1.0}, "nonnegative"),
        ({"eps": 0.0}, "must be positive"),
        ({"annealing_steps": 0}, "must be positive"),
    ],
)
def test_evidential_regression_validates_configuration(kwargs, message):
    """Regression numerical and annealing options are validated explicitly."""
    with pytest.raises(ValueError, match=message):
        EDLRegressionLoss(**kwargs)


@pytest.mark.parametrize("mode", ["evd", "kl", "evd_l2"])
def test_evidential_regression_supports_all_regularizers(mode):
    """Every maintained NIG regularizer supports annealed log-space targets."""
    logits = torch.tensor([[1.0, 1.0, 2.0, 1.0], [2.0, 1.0, 2.0, 1.0]])
    targets = torch.tensor([2.0, 3.0])
    total, nll = EDLRegressionLoss(
        reduction="sum",
        kl_mode=mode,
        annealing_steps=2,
        logspace=True,
    )(logits, targets, iteration=2)
    assert total.ndim == 0
    assert nll.ndim == 0
    assert torch.isfinite(total)


def test_evidential_regression_rejects_nonscalar_targets():
    """NIG regression requires one scalar target per prediction."""
    logits = torch.ones((2, 4))
    with pytest.raises(ValueError, match="targets.shape"):
        EDLRegressionLoss()(logits, torch.ones((2, 1)))


def test_evidential_internal_reducer_rejects_unknown_mode():
    """The shared reducer retains a defensive error for direct callers."""
    with pytest.raises(ValueError, match="Unknown reduction"):
        _reduce_loss(torch.ones(1), "bad")
