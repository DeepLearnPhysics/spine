"""Shared fixtures for image-model tests."""

import pytest

torch = pytest.importorskip("torch")

from spine.data import ClusterLabelBatch, TensorBatch


@pytest.fixture
def image_data():
    """Return two sparse events with cluster, group, shape, and PID labels."""
    data = torch.zeros((7, 7), dtype=torch.float32)
    data[:, 0] = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    data[:, 1:4] = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [4, 0, 0],
            [5, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
        ]
    )
    data[:, 4] = torch.arange(1, 8)
    data[:, 5] = torch.tensor([0, 0, 1, 1, 0, 0, 1])
    data[:, 6] = torch.tensor([0, 0, 1, 1, 0, 0, 1])
    particle_counts = torch.tensor([2, 2])
    particles = {
        "particle": TensorBatch(torch.tensor([0, 1, 0, 1]), particle_counts),
        "group": TensorBatch(torch.tensor([0, 0, 0, 1]), particle_counts),
        "ancestor": TensorBatch(torch.tensor([0, 1, 0, 1]), particle_counts),
        "interaction": TensorBatch(torch.tensor([0, 0, 0, 0]), particle_counts),
        "nu": TensorBatch(torch.tensor([0, 0, 0, 0]), particle_counts),
        "pid": TensorBatch(torch.tensor([2, 3, 1, 4]), particle_counts),
        "group_primary": TensorBatch(torch.ones(4), particle_counts),
        "interaction_primary": TensorBatch(torch.ones(4), particle_counts),
        "vertex": TensorBatch(torch.zeros((4, 3)), particle_counts),
        "momentum": TensorBatch(
            torch.tensor([200.0, 300.0, 10.0, 500.0]), particle_counts
        ),
        "energy_init": TensorBatch(torch.zeros(4), particle_counts),
        "shape": TensorBatch(torch.ones(4), particle_counts),
    }
    return ClusterLabelBatch(
        TensorBatch(data, counts=torch.tensor([4, 3]), has_batch_col=True),
        particles,
    )
