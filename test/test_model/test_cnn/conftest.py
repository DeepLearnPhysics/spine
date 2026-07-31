"""Shared fixtures for sparse CNN layer tests."""

import itertools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("MinkowskiEngine")


@pytest.fixture
def cnn_config():
    """Return a small common CNN configuration."""
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


@pytest.fixture
def sparse_table():
    """Return two small images as a coordinate/feature table."""
    rows = []
    for batch, point in itertools.product(
        range(2), itertools.product(range(4), repeat=3)
    ):
        rows.append((batch, *point, float(sum(point) + 1)))
    return torch.tensor(rows, dtype=torch.float32)
