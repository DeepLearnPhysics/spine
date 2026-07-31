"""Behavioral tests for the PointNet++ encoder."""

import torch
from torch_geometric.data import Batch, Data

from spine.model.pointcloud import PointNetEncoder


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
