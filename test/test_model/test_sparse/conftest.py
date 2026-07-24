"""Shared fixtures for sparse model tests."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("MinkowskiEngine")

from spine.model.uresnet import UResNetSegmentation  # noqa: E402


@pytest.fixture
def uresnet_config():
    """Return a minimal UResNet configuration for sparse contract tests."""
    return {
        "num_classes": 3,
        "reps": 1,
        "depth": 2,
        "filters": 2,
        "num_input": 1,
        "activation": "relu",
        "norm_layer": "none",
    }


@pytest.fixture
def uresnet(uresnet_config):
    """Build a small UResNet segmentation model."""
    return UResNetSegmentation(uresnet_config)
