"""Tests for reusable gradient parameter grouping and diagnostics."""

import math

import pytest

from spine.model.common.gradient import GradientTracker
from spine.utils.conditional import TORCH_AVAILABLE, torch

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")


def test_gradient_tracker_collects_global_and_named_groups_on_cadence():
    """Groups should share parameters while retaining a stable log schema."""
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    inactive = torch.nn.Parameter(torch.tensor(1.0))
    weight.grad = torch.tensor([3.0, 4.0])
    tracker = GradientTracker(
        {
            "network.weight": weight,
            "network.alias": weight,
            "loss.scale": inactive,
        },
        {
            "interval": 2,
            "groups": {"network": "network.*"},
        },
    )

    skipped = tracker.collect(0)
    assert skipped["gradient_sampled"] == 0
    assert math.isnan(skipped["gradient_global_norm"])
    assert len(tracker.groups["network"].parameters) == 1
    assert tracker.groups["network"].parameters[0] is weight

    metrics = tracker.collect(1)
    assert metrics.keys() == skipped.keys()
    assert metrics["gradient_sampled"] == 1
    assert metrics["gradient_global_norm"] == pytest.approx(5.0)
    assert metrics["gradient_global_rms"] == pytest.approx(math.sqrt(12.5))
    assert metrics["gradient_global_abs_max"] == pytest.approx(4.0)
    assert metrics["gradient_global_missing_fraction"] == pytest.approx(0.5)
    assert metrics["gradient_global_nonfinite_count"] == 0
    assert metrics["gradient_network_missing_fraction"] == 0.0


def test_gradient_tracker_handles_sparse_missing_and_nonfinite_gradients():
    """Sparse zeros and malformed gradients should produce explicit metrics."""
    sparse = torch.nn.Parameter(torch.ones(4))
    sparse.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 3]]),
        torch.tensor([3.0, 4.0]),
        size=(4,),
    )
    invalid = torch.nn.Parameter(torch.ones(1))
    invalid.grad = torch.tensor([float("nan")])
    tracker = GradientTracker(
        {"network.sparse": sparse, "network.invalid": invalid},
        {
            "include_global": False,
            "groups": {
                "sparse": "network.sparse",
                "invalid": ["network.invalid"],
            },
        },
    )

    metrics = tracker.collect(0)
    assert metrics["gradient_sparse_norm"] == pytest.approx(5.0)
    assert metrics["gradient_sparse_rms"] == pytest.approx(2.5)
    assert metrics["gradient_sparse_abs_max"] == pytest.approx(4.0)
    assert metrics["gradient_invalid_nonfinite_count"] == 1
    assert math.isnan(metrics["gradient_invalid_norm"])

    sparse.grad = torch.sparse_coo_tensor(
        torch.empty((1, 0), dtype=torch.long),
        torch.empty(0),
        size=(4,),
    )
    assert tracker.collect(0)["gradient_sparse_abs_max"] == 0.0


def test_gradient_tracker_builds_canonical_names_from_modules():
    """Module prefixes should remain stable and frozen parameters be omitted."""
    network = torch.nn.Linear(2, 1)
    network.bias.requires_grad_(False)
    loss = torch.nn.Linear(1, 1, bias=False)
    tracker = GradientTracker.from_modules(
        {"network": network, "unused": None, "loss": loss},
        {
            "groups": {
                "network": "network.*",
                "loss": "loss.*",
            }
        },
    )

    assert list(tracker.named_parameters) == ["network.weight", "loss.weight"]
    assert [name for name, _ in tracker.groups["network"].named_parameters] == [
        "network.weight"
    ]


@pytest.mark.parametrize(
    ("parameters", "config", "error", "message"),
    [
        ({}, 1, TypeError, "boolean or mapping"),
        ({}, False, ValueError, "only when tracking is enabled"),
        ({}, {"unknown": True}, TypeError, "Unexpected"),
        ({}, {"interval": 1.5}, TypeError, "interval.*integer"),
        ({}, {"interval": True}, TypeError, "interval.*integer"),
        ({}, {"interval": 0}, ValueError, "interval.*positive"),
        ({}, {"include_global": "yes"}, TypeError, "include_global.*boolean"),
        ({}, {"groups": []}, TypeError, "groups.*mapping"),
        ({}, True, ValueError, "requires trainable parameters"),
    ],
)
def test_gradient_tracker_rejects_invalid_top_level_configuration(
    parameters, config, error, message
):
    """Malformed tracker settings should fail before training starts."""
    with pytest.raises(error, match=message):
        GradientTracker(parameters, config)


@pytest.mark.parametrize(
    ("groups", "error", "message"),
    [
        ({"bad-name": "network.*"}, ValueError, "group names"),
        ({"global": "network.*"}, ValueError, "reserved"),
        ({"head": 3}, TypeError, "pattern or sequence"),
        ({"head": []}, ValueError, "non-empty strings"),
        ({"head": [""]}, ValueError, "non-empty strings"),
        ({"head": "loss.*"}, ValueError, "matched no"),
    ],
)
def test_gradient_tracker_rejects_invalid_group_configuration(groups, error, message):
    """Every requested group should be safe to log and match parameters."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(error, match=message):
        GradientTracker({"network.weight": parameter}, {"groups": groups})

    if "head" in groups:
        with pytest.raises(error, match=message):
            GradientTracker(
                {"network.weight": parameter},
                {"include_global": False, "groups": groups},
            )


def test_gradient_tracker_requires_at_least_one_enabled_group():
    """Disabling the implicit group requires an explicit replacement."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="at least one configured group"):
        GradientTracker(
            {"network.weight": parameter},
            {"include_global": False},
        )


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        (lambda: [], TypeError, "mapping"),
        (
            lambda: {"": torch.nn.Parameter(torch.tensor(1.0))},
            ValueError,
            "non-empty",
        ),
        (lambda: {"weight": torch.tensor(1.0)}, TypeError, "not a parameter"),
    ],
)
def test_gradient_tracker_validates_parameter_registry(parameters, error, message):
    """The shared registry should reject ambiguous parameter declarations."""
    with pytest.raises(error, match=message):
        GradientTracker(parameters())


def test_gradient_tracker_validates_module_registry_and_iterations():
    """Module construction and collection require explicit stable identities."""
    with pytest.raises(TypeError, match="modules.*mapping"):
        GradientTracker.from_modules([])
    with pytest.raises(ValueError, match="module names"):
        GradientTracker.from_modules({"": torch.nn.Linear(1, 1)})
    with pytest.raises(TypeError, match="must be a module"):
        GradientTracker.from_modules({"network": object()})

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    tracker = GradientTracker({"network.weight": parameter})
    with pytest.raises(TypeError, match="iteration.*integer"):
        tracker.collect(True)
    with pytest.raises(ValueError, match="iteration.*nonnegative"):
        tracker.collect(-1)

    metrics = tracker.collect(0)
    assert metrics["gradient_global_norm"] == 0.0
    assert metrics["gradient_global_rms"] == 0.0
    assert metrics["gradient_global_abs_max"] == 0.0
    assert metrics["gradient_global_missing_fraction"] == 1.0
