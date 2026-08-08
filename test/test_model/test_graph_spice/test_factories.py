"""Tests for clustering-component factories."""

import pytest

from spine.model.cnn.fpn import FPN
from spine.model.cnn.uresnet_layers import UResNet
from spine.model.graph_spice import (
    EdgeLoss,
    kernel_factory,
    loss_factory,
)
from spine.model.graph_spice.kernel import BilinearKernel


def test_kernel_factory_builds_supported_kernel():
    """The kernel factory must resolve maintained configuration names."""
    config = {"name": "bilinear", "num_features": 8}

    kernel = kernel_factory(config)

    assert isinstance(kernel, BilinearKernel)
    assert config == {"name": "bilinear", "num_features": 8}


def test_loss_factory_builds_supported_loss():
    """The loss factory must resolve maintained configuration names."""
    loss = loss_factory({"name": "edge", "metric": None})

    assert isinstance(loss, EdgeLoss)


@pytest.mark.parametrize(
    ("name", "model_type"),
    [("uresnet", UResNet), ("fpn", FPN)],
)
def test_backbone_factory_uses_shared_cnn_modules(name, model_type, cnn_config):
    """The GraphSPICE backbone factory resolves shared CNN implementations."""
    from spine.model.graph_spice import backbone_factory

    model = backbone_factory({"name": name, **cnn_config})

    assert isinstance(model, model_type)


@pytest.mark.parametrize(
    ("factory", "name"),
    [
        (kernel_factory, "mixed"),
        (kernel_factory, "retired_kernel"),
        (loss_factory, "retired_loss"),
    ],
)
def test_cluster_factories_reject_unsupported_components(factory, name):
    """Factories must not expose unmaintained implementation details."""
    with pytest.raises(ValueError, match=name):
        factory(name)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("uresnet", "configuration block"),
        ({}, "requires a `name`"),
        ({"name": "transformer"}, "Unknown backbone"),
    ],
)
def test_backbone_factory_validates_named_configuration(config, message):
    """Backbones require a complete, registered CNN configuration block."""
    from spine.model.graph_spice import backbone_factory

    with pytest.raises(ValueError, match=message):
        backbone_factory(config)
