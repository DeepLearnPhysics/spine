"""Tests for task-gradient projection strategies."""

import math
from types import SimpleNamespace

import pytest

from spine.model.common.gradient import ParameterGroup
from spine.model.common.gradient_surgery import PCGrad
from spine.model.common.loss_balancing import LossTerm
from spine.utils.conditional import TORCH_AVAILABLE, torch

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")


def make_group(**parameters):
    """Build a canonical parameter group from keyword arguments."""
    return ParameterGroup(
        "pcgrad",
        tuple((f"network.{name}", parameter) for name, parameter in parameters.items()),
    )


def test_pcgrad_projects_conflicting_gradients_and_reports_diagnostics():
    """Conflicting task components should be removed before optimization."""
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    first = parameter[0]
    second = -parameter[0] + parameter[1]
    strategy = PCGrad(make_group(weight=parameter), seed=7)

    metrics = strategy.backward(
        first + second,
        {
            "first": LossTerm(first, "composite"),
            "second": LossTerm(second, "composite"),
        },
        iteration=3,
    )

    torch.testing.assert_close(parameter.grad, torch.tensor([0.5, 1.5]))
    assert metrics["pcgrad_active_tasks"] == 2
    assert metrics["pcgrad_shared_parameters"] == 1
    assert metrics["pcgrad_projection_count"] == 2
    assert metrics["pcgrad_first_gradient_norm"] == pytest.approx(1.0)
    assert metrics["pcgrad_second_gradient_norm"] == pytest.approx(math.sqrt(2.0))
    assert metrics["pcgrad_mean_cosine"] == pytest.approx(-1.0 / math.sqrt(2.0))
    assert metrics["pcgrad_conflict_fraction"] == 1.0


def test_pcgrad_preserves_constructive_and_task_specific_gradients():
    """Non-conflicting and unshared parameters should retain ordinary gradients."""
    shared = torch.nn.Parameter(torch.tensor(1.0))
    other_shared = torch.nn.Parameter(torch.tensor(2.0))
    constructive = PCGrad(make_group(shared=shared, other_shared=other_shared))
    first = shared + other_shared
    second = 2.0 * shared + 2.0 * other_shared
    metrics = constructive.backward(
        first + second,
        {
            "first": LossTerm(first, "composite"),
            "second": LossTerm(second, "composite"),
        },
        0,
    )
    torch.testing.assert_close(shared.grad, torch.tensor(3.0))
    torch.testing.assert_close(other_shared.grad, torch.tensor(3.0))
    assert metrics["pcgrad_projection_count"] == 0
    assert metrics["pcgrad_conflict_fraction"] == 0.0

    first_parameter = torch.nn.Parameter(torch.tensor(1.0))
    second_parameter = torch.nn.Parameter(torch.tensor(1.0))
    separate = PCGrad(make_group(first=first_parameter, second=second_parameter))
    first = first_parameter.square()
    second = second_parameter.square()
    metrics = separate.backward(
        first + second,
        {
            "first": LossTerm(first, "composite"),
            "second": LossTerm(second, "composite"),
        },
        0,
    )
    torch.testing.assert_close(first_parameter.grad, torch.tensor(2.0))
    torch.testing.assert_close(second_parameter.grad, torch.tensor(2.0))
    assert metrics["pcgrad_shared_parameters"] == 0
    assert math.isnan(metrics["pcgrad_mean_cosine"])


def test_pcgrad_replaces_only_shared_candidates_during_projection():
    """A projected update should retain ordinary task-specific gradients."""
    shared = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    task_specific = torch.nn.Parameter(torch.tensor(2.0))
    first = shared[0] + task_specific.square()
    second = -shared[0] + shared[1]
    strategy = PCGrad(make_group(shared=shared, task_specific=task_specific))

    metrics = strategy.backward(
        first + second,
        {
            "first": LossTerm(first, "composite"),
            "second": LossTerm(second, "composite"),
        },
        0,
    )

    torch.testing.assert_close(shared.grad, torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(task_specific.grad, torch.tensor(4.0))
    assert metrics["pcgrad_shared_parameters"] == 1


def test_pcgrad_applies_scales_and_ignores_inactive_objectives():
    """Producer priorities and activity metadata should define task gradients."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    first = parameter
    second = parameter
    inactive = -parameter
    strategy = PCGrad(make_group(weight=parameter))
    metrics = strategy.backward(
        2.0 * first + second,
        {
            "first": LossTerm(first, "composite", scale=2.0),
            "second": LossTerm(second, "composite"),
            "inactive": LossTerm(inactive, "composite", active=False),
        },
        0,
    )

    torch.testing.assert_close(parameter.grad, torch.tensor(3.0))
    assert metrics["pcgrad_active_tasks"] == 2
    assert metrics["pcgrad_first_gradient_norm"] == 2.0
    assert metrics["pcgrad_second_gradient_norm"] == 1.0
    assert math.isnan(metrics["pcgrad_inactive_gradient_norm"])


def test_pcgrad_handles_active_objective_without_network_dependency():
    """An active loss-module-only objective should leave network gradients alone."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    network_term = parameter.square()
    detached_term = torch.tensor(2.0)
    strategy = PCGrad(make_group(weight=parameter))

    metrics = strategy.backward(
        network_term,
        {
            "network": LossTerm(network_term, "composite"),
            "detached": LossTerm(detached_term, "composite"),
        },
        0,
    )

    torch.testing.assert_close(parameter.grad, torch.tensor(2.0))
    assert metrics["pcgrad_detached_gradient_norm"] == 0.0


def test_pcgrad_is_deterministic_per_iteration_with_three_tasks():
    """Projection order should be reproducible without consuming global RNG."""
    parameter = torch.nn.Parameter(torch.ones(2))
    gradients = {
        "a": (torch.tensor([1.0, 0.0]),),
        "b": (torch.tensor([-1.0, 1.0]),),
        "c": (torch.tensor([0.5, -2.0]),),
    }
    strategy = PCGrad(make_group(weight=parameter), seed=11)

    first, first_count = strategy._project(gradients, (True,), iteration=4)
    second, second_count = strategy._project(gradients, (True,), iteration=4)

    assert first_count == second_count
    for name in gradients:
        torch.testing.assert_close(first[name][0], second[name][0])


def test_pcgrad_skips_projection_onto_zero_gradient():
    """Zero-gradient tasks should not introduce an undefined projection."""
    parameter = torch.nn.Parameter(torch.ones(2))
    gradients = {
        "zero": (torch.zeros(2),),
        "nonzero": (torch.tensor([1.0, -1.0]),),
    }
    strategy = PCGrad(make_group(weight=parameter))

    projected, count = strategy._project(gradients, (True,), iteration=0)

    torch.testing.assert_close(projected["zero"][0], torch.zeros(2))
    torch.testing.assert_close(projected["nonzero"][0], gradients["nonzero"][0])
    assert count == 0


def test_pcgrad_constructs_from_canonical_network_patterns():
    """PCGrad should reuse the monitor's strict canonical parameter registry."""
    network = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    strategy = PCGrad.from_module(
        network,
        {
            "name": "PCGrad",
            "parameters": ["network.0.*"],
            "seed": 9,
        },
    )

    assert strategy.seed == 9
    assert all(
        name.startswith("network.0.")
        for name, _ in strategy.parameter_group.named_parameters
    )


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ([], TypeError, "configuration.*mapping"),
        ({}, ValueError, "strategy `name`"),
        ({"name": 3}, ValueError, "only `pcgrad`"),
        ({"name": "gradnorm"}, ValueError, "only `pcgrad`"),
        ({"name": "pcgrad", "parameters": 3}, TypeError, "glob pattern"),
        ({"name": "pcgrad", "unknown": True}, TypeError, "Unexpected"),
        ({"name": "pcgrad", "seed": True}, TypeError, "seed.*integer"),
        (
            {"name": "pcgrad", "parameters": "network.missing.*"},
            ValueError,
            "matched no",
        ),
    ],
)
def test_pcgrad_rejects_invalid_configuration(config, error, message):
    """Configuration errors should fail before the first training batch."""
    with pytest.raises(error, match=message):
        PCGrad.from_module(torch.nn.Linear(1, 1), config)


def test_pcgrad_validates_parameter_group():
    """Direct construction should require a non-empty shared group."""
    with pytest.raises(TypeError, match="ParameterGroup"):
        PCGrad([])
    with pytest.raises(ValueError, match="at least one"):
        PCGrad(ParameterGroup("empty", ()))


@pytest.mark.parametrize(
    ("loss", "terms", "iteration", "error", "message"),
    [
        (
            lambda p: torch.ones(2, requires_grad=True),
            lambda p: {"task": LossTerm(p, "composite")},
            0,
            TypeError,
            "scalar loss",
        ),
        (
            lambda p: p,
            lambda p: [],
            0,
            TypeError,
            "terms.*mapping",
        ),
        (lambda p: p, lambda p: {}, 0, ValueError, "at least one"),
        (
            lambda p: p,
            lambda p: {"task": LossTerm(p, "composite")},
            True,
            TypeError,
            "iteration.*integer",
        ),
        (
            lambda p: p,
            lambda p: {"task": LossTerm(p, "composite")},
            -1,
            ValueError,
            "iteration.*nonnegative",
        ),
        (
            lambda p: p,
            lambda p: {"": LossTerm(p, "composite")},
            0,
            ValueError,
            "objective names",
        ),
        (
            lambda p: p,
            lambda p: {
                "task": SimpleNamespace(value=torch.ones(2), scale=1.0, active=True)
            },
            0,
            TypeError,
            "scalar tensor",
        ),
        (
            lambda p: p,
            lambda p: {"task": SimpleNamespace(value=p, scale="one", active=True)},
            0,
            ValueError,
            "invalid scale",
        ),
        (
            lambda p: p,
            lambda p: {"task": SimpleNamespace(value=p, scale=-1.0, active=True)},
            0,
            ValueError,
            "nonnegative",
        ),
        (
            lambda p: p,
            lambda p: {"task": SimpleNamespace(value=p, scale=1.0)},
            0,
            TypeError,
            "activity flag",
        ),
    ],
)
def test_pcgrad_rejects_invalid_runtime_metadata(
    loss, terms, iteration, error, message
):
    """Malformed objective metadata should fail before gradients are changed."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    strategy = PCGrad(make_group(weight=parameter))
    with pytest.raises(error, match=message):
        strategy.backward(loss(parameter), terms(parameter), iteration)


def test_pcgrad_rejects_sparse_selected_gradients():
    """Unsupported sparse surgery should fail explicitly instead of densifying."""
    embedding = torch.nn.Embedding(4, 2, sparse=True)
    parameter = embedding.weight
    value = embedding(torch.tensor([0])).sum()
    strategy = PCGrad(make_group(weight=parameter))

    with pytest.raises(TypeError, match="dense gradients"):
        strategy.backward(
            value,
            {"task": LossTerm(value, "composite")},
            0,
        )
