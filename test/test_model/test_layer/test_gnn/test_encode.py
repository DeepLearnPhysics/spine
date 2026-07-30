"""Behavioral tests for GNN feature encoders."""

import numpy as np
import torch

from spine.data import IndexBatch, TensorBatch
from spine.model.layer.gnn.encode import (
    ClustGeoCNNMixNodeEncoder,
    ClustGeoNodeEncoder,
)


def test_torch_geometric_encoder_handles_degenerate_cluster():
    data = TensorBatch(
        torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 2.0],
                [0.0, 1.0, 1.0, 1.0, 3.0],
            ]
        ),
        counts=[2],
    )
    clusters = IndexBatch(
        [np.array([0, 1], dtype=np.int64)],
        spans=[2],
        counts=[1],
        single_counts=[2],
    )

    features = ClustGeoNodeEncoder(use_numpy=False)(
        data,
        clusters,
    ).torch_tensor()

    assert features.shape == (1, 16)
    assert torch.isfinite(features).all()


def test_mixed_node_encoder_combines_numpy_and_torch_features(
    graph_data,
    graph_clusters,
):
    encoder = ClustGeoCNNMixNodeEncoder(
        geo_encoder={},
        cnn_encoder={
            "reps": 1,
            "depth": 1,
            "filters": 4,
            "num_input": 1,
            "data_dim": 3,
            "activation": "relu",
            "norm_layer": "none",
            "spatial_size": 4,
            "feature_size": 6,
        },
    )

    output = encoder(graph_data.to_tensor(), graph_clusters)

    assert output.shape == (3, 22)
    assert not output.is_numpy
