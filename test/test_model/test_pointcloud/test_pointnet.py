"""Behavioral tests for the PointNet++ encoder."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from spine.model.pointcloud import PointNet, PointNetEncoder


def test_pointnet_encoder_runs_batched_point_clouds():
    encoder = PointNetEncoder(
        {
            "pointnet": {
                "depth": 1,
                "sampling_ratio": 1.0,
                "neighbor_radius": 10.0,
                "mlp_specs_0": [4, 8],
                "mlp_specs_glob": [11, 8],
                "mlp_specs_final": [8, 4],
                "dropout": 0.0,
            }
        }
    )
    batch = Batch.from_data_list(
        [
            Data(
                x=torch.ones((3, 1)),
                pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ),
            Data(
                x=torch.ones((3, 1)),
                pos=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            ),
        ]
    )

    output = encoder(batch)

    assert output.shape == (2, 4)
    assert encoder.feature_size == 4


def _pointnet_config(**updates):
    """Return the smallest complete PointNet architecture configuration."""
    config = {
        "depth": 1,
        "sampling_ratio": 1.0,
        "neighbor_radius": 10.0,
        "mlp_specs_0": [4, 8],
        "mlp_specs_glob": [11, 8],
        "mlp_specs_final": [8, 4],
        "dropout": 0.0,
    }
    config.update(updates)
    return {"pointnet": config}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "missing `pointnet`"),
        (_pointnet_config(depth=0), "depth"),
        (_pointnet_config(sampling_ratio=[1.0, 0.5]), "sampling ratio per"),
        (_pointnet_config(sampling_ratio="all"), "number or a list"),
        (_pointnet_config(sampling_ratio=0.0), "lie in"),
        (_pointnet_config(neighbor_radius=[1.0, 2.0]), "neighbor radius per"),
        (_pointnet_config(neighbor_radius="near"), "number or a list"),
        (_pointnet_config(neighbor_radius=0.0), "must be positive"),
    ],
)
def test_pointnet_validates_architecture(config, message):
    """Malformed hierarchy parameters fail before constructing PyG modules."""
    with pytest.raises(ValueError, match=message):
        PointNet(config)


def test_pointnet_requires_every_mlp_and_batch_assignment():
    """Each abstraction level and every forward input need explicit context."""
    config = _pointnet_config()
    del config["pointnet"]["mlp_specs_0"]
    with pytest.raises(ValueError, match="mlp_specs_0"):
        PointNet(config)

    model = PointNet(_pointnet_config())
    cloud = Data(x=torch.ones((2, 1)), pos=torch.zeros((2, 3)))
    with pytest.raises(ValueError, match="batch IDs"):
        model(cloud)


def test_pointnet_accepts_per_level_sampling_and_radius_lists():
    """Hierarchies may configure sampling and neighborhood scales per level."""
    config = _pointnet_config(
        depth=2,
        sampling_ratio=[1.0, 0.5],
        neighbor_radius=[10.0, 20.0],
        mlp_specs_1=[11, 8],
        mlp_specs_glob=[11, 8],
    )
    model = PointNet(config)

    assert model.sampling_ratios == [1.0, 0.5]
    assert model.neighbor_radii == [10.0, 20.0]
