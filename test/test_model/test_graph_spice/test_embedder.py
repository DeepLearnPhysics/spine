"""Behavioral tests for the GraphSPICE feature embedder."""

import itertools

import pytest
import torch

from spine.data import TensorBatch
from spine.model.graph_spice import GraphSPICEEmbedder


def small_uresnet_config():
    """Return a minimal UResNet configuration for embedder tests."""
    return {
        "reps": 1,
        "depth": 2,
        "filters": 4,
        "num_input": 4,
        "data_dim": 3,
        "activation": "relu",
        "norm_layer": "none",
        "spatial_size": 4,
    }


def point_cloud_batch():
    """Return one dense 4-cube represented as a sparse point table."""
    rows = []
    for point in itertools.product(range(4), repeat=3):
        rows.append((0, *point, float(sum(point) + 1)))
    tensor = torch.tensor(rows, dtype=torch.float32)
    return TensorBatch(tensor, counts=[len(tensor)], has_batch_col=True)


def test_embedder_hypergraph_uses_published_spatial_embeddings():
    """Kernel input and named output must share absolute spatial embeddings."""
    embedder = GraphSPICEEmbedder(
        small_uresnet_config(),
        feature_embedding_dim=2,
        spatial_embedding_dim=3,
        use_raw_features=False,
    )

    result = embedder(point_cloud_batch())

    spatial_embeddings = result["spatial_embeddings"].torch_tensor()
    hypergraph_features = result["hypergraph_features"].torch_tensor()
    torch.testing.assert_close(hypergraph_features[:, :3], spatial_embeddings)


def test_embedder_rejects_incompatible_spatial_dimension():
    """Spatial offsets must have the same dimension as input coordinates."""
    with pytest.raises(ValueError, match="must match the input dimension"):
        GraphSPICEEmbedder(
            small_uresnet_config(),
            spatial_embedding_dim=2,
            use_raw_features=False,
        )
