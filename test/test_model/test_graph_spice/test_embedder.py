"""Behavioral tests for the GraphSPICE feature embedder."""

import itertools

import numpy as np
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
    return TensorBatch(
        tensor,
        counts=[len(tensor)],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )


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


def test_embedder_uses_primary_feature_from_multifeature_input():
    """Auxiliary point features should not make the charge input ambiguous."""
    embedder = GraphSPICEEmbedder(
        small_uresnet_config(),
        feature_embedding_dim=2,
        spatial_embedding_dim=3,
        use_raw_features=False,
    )
    data = point_cloud_batch()
    rows = torch.cat((data.data, torch.full((len(data.data), 2), 1000.0)), dim=1)
    multifeature = TensorBatch(
        rows,
        data.counts,
        has_batch_col=data.has_batch_col,
        coord_cols=data.coord_cols,
    )

    result = embedder(multifeature)

    assert result["features"].shape[0] == len(data.data)
    assert result["hypergraph_features"].shape == (len(data.data), 8)


def test_embedder_rejects_incompatible_spatial_dimension():
    """Spatial offsets must have the same dimension as input coordinates."""
    with pytest.raises(ValueError, match="must match the input dimension"):
        GraphSPICEEmbedder(
            small_uresnet_config(),
            spatial_embedding_dim=2,
            use_raw_features=False,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"feature_embedding_dim": 0}, "feature_embedding_dim"),
        ({"spatial_embedding_dim": 0}, "spatial_embedding_dim"),
        ({"covariance_mode": "linear"}, "Covariance mode"),
        ({"occupancy_mode": "linear"}, "Occupancy mode"),
        ({"predict_semantics": True}, "number of classes"),
    ],
)
def test_embedder_validates_output_configuration(updates, message):
    """Embedding dimensions, activations, and semantic heads are explicit."""
    with pytest.raises(ValueError, match=message):
        GraphSPICEEmbedder(small_uresnet_config(), **updates)


def test_embedder_requires_positive_spatial_extent():
    """Coordinate normalization requires a configured positive image size."""
    config = small_uresnet_config()
    config.pop("spatial_size")
    with pytest.raises(ValueError, match="spatial size"):
        GraphSPICEEmbedder(config)

    config["spatial_size"] = 0
    with pytest.raises(ValueError, match="spatial_size"):
        GraphSPICEEmbedder(config)


def test_embedder_supports_raw_features_semantics_and_exp_activations():
    """Raw and projected output modes expose their documented products."""
    raw = GraphSPICEEmbedder(
        small_uresnet_config(),
        use_raw_features=True,
        coord_conv=True,
        predict_semantics=True,
        num_classes=3,
        covariance_mode="exp",
        occupancy_mode="exp",
    )(point_cloud_batch())
    assert set(raw) == {"coordinates", "features", "segmentation"}
    assert raw["segmentation"].shape == (64, 3)

    config = small_uresnet_config()
    config["num_input"] = 1
    projected = GraphSPICEEmbedder(
        config,
        coord_conv=False,
        feature_embedding_dim=2,
    )(point_cloud_batch())
    assert projected["hypergraph_features"].shape == (64, 8)
