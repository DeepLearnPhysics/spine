"""Tests for model-owned optimizer implementations and factories."""

import pytest
import torch

from spine.model.optim import factory as optim_module
from spine.model.optim import lr_sched_factory, optim_factory
from spine.model.optim.adabound import AdaBound, AdaBoundW


@pytest.mark.parametrize("optimizer_cls", [AdaBound, AdaBoundW])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": -1.0}, "learning rate"),
        ({"eps": -1.0}, "epsilon"),
        ({"betas": (-0.1, 0.9)}, "index 0"),
        ({"betas": (0.9, 1.0)}, "index 1"),
        ({"final_lr": -1.0}, "final learning"),
        ({"gamma": 1.0}, "gamma"),
    ],
)
def test_adabound_validates_hyperparameters(optimizer_cls, kwargs, message):
    """Both AdaBound variants should reject invalid scalar settings."""
    with pytest.raises(ValueError, match=message):
        optimizer_cls([torch.nn.Parameter(torch.ones(1))], **kwargs)


@pytest.mark.parametrize("optimizer_cls", [AdaBound, AdaBoundW])
@pytest.mark.parametrize("amsbound", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_adabound_steps_all_algorithm_variants(optimizer_cls, amsbound, weight_decay):
    """Optimizer steps should initialize state, update parameters, and return closures."""
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = optimizer_cls(
        [parameter],
        lr=0.01,
        final_lr=0.1,
        gamma=0.1,
        amsbound=amsbound,
        weight_decay=weight_decay,
    )
    assert optimizer.step(lambda: "no-gradient") == "no-gradient"
    parameter.grad = torch.tensor([0.5, -0.25])
    before = parameter.detach().clone()
    assert optimizer.step(lambda: "loss") == "loss"
    assert not torch.equal(parameter, before)
    assert optimizer.state[parameter]["step"] == 1
    if amsbound:
        assert "max_exp_avg_sq" in optimizer.state[parameter]

    state = optimizer.__dict__.copy()
    optimizer.param_groups[0].pop("amsbound")
    optimizer.__setstate__(state)
    assert optimizer.param_groups[0]["amsbound"] is False


@pytest.mark.parametrize("optimizer_cls", [AdaBound, AdaBoundW])
def test_adabound_rejects_sparse_gradients(optimizer_cls):
    """Dense AdaBound implementations must reject sparse gradients clearly."""
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = optimizer_cls([parameter])
    parameter.grad = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([1.0]), size=(2,)
    )
    with pytest.raises(RuntimeError, match="sparse gradients"):
        optimizer.step()


def test_optimizer_and_scheduler_factories(monkeypatch):
    """Factories should expose custom/default optimizers and optional-runtime errors."""
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = optim_factory({"name": "SGD", "lr": 0.1}, [parameter])
    scheduler = lr_sched_factory(
        {"name": "StepLR", "step_size": 1},
        optimizer,
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    monkeypatch.setattr(optim_module, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError, match="optimizer"):
        optim_module.optim_dict()
    with pytest.raises(ImportError, match="scheduler"):
        optim_module.lr_sched_factory({}, optimizer)
