"""Shared fixtures for GraphSPICE component tests."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("MinkowskiEngine")


@pytest.fixture
def cnn_config():
    """Return a small shared-CNN configuration."""
    return {
        "reps": 1,
        "depth": 2,
        "filters": 4,
        "num_input": 1,
        "data_dim": 3,
        "activation": "relu",
        "norm_layer": "none",
        "spatial_size": 4,
    }
