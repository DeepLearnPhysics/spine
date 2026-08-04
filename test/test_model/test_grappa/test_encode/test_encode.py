"""Behavioral tests for GNN feature encoders."""

import numpy as np
import pytest
import torch

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.model.grappa.encode import (
    ClustCNNEdgeEncoder,
    ClustCNNGlobalEncoder,
    ClustCNNNodeEncoder,
    ClustGeoCNNMixEdgeEncoder,
    ClustGeoCNNMixNodeEncoder,
    ClustGeoEdgeEncoder,
    ClustGeoNodeEncoder,
    EmptyClusterEdgeEncoder,
    EmptyClusterGlobalEncoder,
    EmptyClusterNodeEncoder,
)
from spine.model.grappa.graph import CompleteGraph

CNN_CONFIG = {
    "reps": 1,
    "depth": 1,
    "filters": 4,
    "num_input": 1,
    "data_dim": 3,
    "activation": "relu",
    "norm_layer": "none",
    "spatial_size": 4,
    "feature_size": 6,
}


def test_torch_geometric_encoder_handles_degenerate_cluster():
    data = TensorBatch(
        torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 2.0],
                [0.0, 1.0, 1.0, 1.0, 3.0],
            ]
        ),
        counts=[2],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
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
        cnn_encoder=CNN_CONFIG,
    )

    output = encoder(graph_data.to_tensor(), graph_clusters)

    assert output.shape == (3, 22)
    assert not output.is_numpy


def test_empty_encoders_preserve_graph_batching(graph_data, graph_clusters):
    """Empty encoders emit zero-width features with canonical item counts."""
    data = graph_data.to_tensor()
    edge_index, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)

    nodes = EmptyClusterNodeEncoder()(data, graph_clusters)
    edges = EmptyClusterEdgeEncoder()(data, graph_clusters, edge_index)
    globals_ = EmptyClusterGlobalEncoder()(data, graph_clusters)

    assert nodes.shape == (3, 0)
    assert nodes.counts.tolist() == [2, 1]
    assert edges.shape == (1, 0)
    assert edges.counts.tolist() == [1, 0]
    assert globals_.shape == (2, 0)
    assert globals_.counts.tolist() == [1, 1]


def test_cnn_encoders_cover_node_edge_and_global_graph_views(
    graph_data,
    graph_clusters,
):
    """CNN encoders construct images at all three graph aggregation levels."""
    data = graph_data.to_tensor()
    directed, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)
    undirected, _, _ = CompleteGraph(directed=False)(graph_data, graph_clusters)

    nodes = ClustCNNNodeEncoder(**CNN_CONFIG)(data, graph_clusters)
    edges = ClustCNNEdgeEncoder(**CNN_CONFIG)(data, graph_clusters, undirected)
    globals_ = ClustCNNGlobalEncoder(**CNN_CONFIG)(data, graph_clusters)

    assert nodes.shape == (3, 6)
    assert edges.shape == (2, 6)
    assert edges.counts.tolist() == [2, 0]
    assert globals_.shape == (2, 6)

    empty_edges = ClustCNNEdgeEncoder(**CNN_CONFIG)(
        data,
        graph_clusters,
        EdgeIndexBatch(
            torch.empty((2, 0), dtype=torch.long),
            counts=[0, 0],
            spans=graph_clusters.counts,
            directed=True,
        ),
    )
    assert empty_edges.shape == (0, 6)


def test_mixed_edge_encoder_combines_geometry_and_cnn(
    graph_data,
    graph_clusters,
):
    """Mixed edge encoding aligns geometric and learned representations."""
    edge_index, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)
    encoder = ClustGeoCNNMixEdgeEncoder(
        geo_encoder={"use_numpy": False},
        cnn_encoder=CNN_CONFIG,
    )

    output = encoder(graph_data.to_tensor(), graph_clusters, edge_index)

    assert output.shape == (1, 25)
    assert not output.is_numpy


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dir_max_dist": "bad"}, "only take the value 'optimize'"),
        ({"add_local_dirs": True}, "must also add points"),
        ({"add_local_dedxs": True}, "must also add points"),
    ],
)
def test_geometric_node_encoder_validates_dependent_options(kwargs, message):
    """Endpoint-derived quantities require a valid point configuration."""
    with pytest.raises(ValueError, match=message):
        ClustGeoNodeEncoder(**kwargs)


def test_geometric_node_encoder_validates_explicit_inputs(
    graph_data,
    graph_clusters,
):
    """Explicit point and extra tensors must agree with enabled features."""
    data = graph_data.to_tensor()
    points = TensorBatch(torch.zeros((3, 6)), graph_clusters.counts)

    with pytest.raises(ValueError, match="add_points.*True"):
        ClustGeoNodeEncoder(use_numpy=False)(data, graph_clusters, points=points)
    with pytest.raises(ValueError, match="add_value.*add_shape"):
        ClustGeoNodeEncoder(use_numpy=False)(
            data,
            graph_clusters,
            extra=TensorBatch(torch.zeros((3, 1)), graph_clusters.counts),
        )
    with pytest.raises(ValueError, match="either `coord_label` or `points`"):
        ClustGeoNodeEncoder(use_numpy=False, add_points=True)(data, graph_clusters)
    with pytest.raises(TypeError, match="structured cluster labels"):
        ClustGeoNodeEncoder(use_numpy=False, add_shape=True)(data, graph_clusters)


@pytest.mark.parametrize(
    ("kwargs", "width"),
    [
        ({"add_value": True}, 2),
        ({"add_shape": True}, 1),
        ({"add_value": True, "add_shape": True}, 3),
    ],
)
def test_geometric_node_encoder_accepts_aligned_extra_features(
    graph_data,
    graph_clusters,
    kwargs,
    width,
):
    """Precomputed scalar features bypass duplicate geometric extraction."""
    extra = TensorBatch(torch.ones((3, width)), graph_clusters.counts)
    output = ClustGeoNodeEncoder(use_numpy=False, **kwargs)(
        graph_data.to_tensor(),
        graph_clusters,
        extra=extra,
    )
    assert output.shape == (3, 16 + width)

    bad = TensorBatch(torch.ones((3, width + 1)), graph_clusters.counts)
    with pytest.raises(ValueError, match="extra.shape"):
        ClustGeoNodeEncoder(use_numpy=False, **kwargs)(
            graph_data.to_tensor(),
            graph_clusters,
            extra=bad,
        )


def test_geometric_node_encoder_adds_points_directions_and_dedx(
    graph_data,
    graph_clusters,
):
    """Explicit endpoints support every endpoint-derived geometric feature."""
    points = TensorBatch(
        torch.tensor(
            [
                [0, 0, 0, 0, 1, 0],
                [3, 0, 0, 3, 1, 0],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=torch.float32,
        ),
        graph_clusters.counts,
    )
    encoder = ClustGeoNodeEncoder(
        use_numpy=False,
        add_points=True,
        add_local_dirs=True,
        dir_max_dist="optimize",
        add_local_dedxs=True,
    )

    features, returned_points = encoder(
        graph_data.to_tensor(),
        graph_clusters,
        points=points,
    )

    assert returned_points is points
    assert features.shape == (3, 30)
    assert torch.isfinite(features.torch_tensor()).all()


def test_torch_geometric_edge_encoder_handles_reciprocals_and_empty_graph(
    graph_data,
    graph_clusters,
):
    """Torch edge geometry mirrors reciprocal edges and types empty output."""
    data = graph_data.to_tensor()
    undirected, _, _ = CompleteGraph(directed=False)(graph_data, graph_clusters)
    encoder = ClustGeoEdgeEncoder(use_numpy=False)

    features = encoder(data, graph_clusters, undirected)

    assert features.shape == (2, 19)
    torch.testing.assert_close(
        features.torch_tensor()[0, :3],
        features.torch_tensor()[1, 3:6],
    )
    torch.testing.assert_close(
        features.torch_tensor()[0, 6:9],
        -features.torch_tensor()[1, 6:9],
    )

    empty = EdgeIndexBatch(
        torch.empty((2, 0), dtype=torch.long),
        counts=[0, 0],
        spans=graph_clusters.counts,
        directed=True,
    )
    assert encoder(data, graph_clusters, empty).shape == (0, 19)


def test_torch_geometric_node_encoder_covers_scalar_and_empty_clusters(
    graph_labels,
    graph_clusters,
):
    """Torch geometry includes deposition statistics, shape and empty output."""
    encoder = ClustGeoNodeEncoder(
        use_numpy=False,
        add_value=True,
        add_shape=True,
    )

    features = encoder(graph_labels, graph_clusters)

    assert features.shape == (3, 19)
    assert torch.isfinite(features.torch_tensor()).all()

    singleton_clusters = IndexBatch(
        [np.array([0]), np.array([2]), np.array([4])],
        spans=graph_labels.counts,
        counts=[2, 1],
        single_counts=[1, 1, 1],
    )
    singleton_features = encoder(graph_labels, singleton_clusters)
    assert singleton_features.shape == (3, 19)

    empty_clusters = IndexBatch(
        [],
        spans=graph_labels.counts,
        counts=[0, 0],
        single_counts=[],
    )
    assert ClustGeoNodeEncoder(use_numpy=False)(
        graph_labels,
        empty_clusters,
    ).shape == (0, 16)


def test_torch_geometric_edge_encoder_uses_precomputed_closest_pair(
    graph_data,
    graph_clusters,
    monkeypatch,
):
    """Precomputed closest indexes bypass pairwise distance calculation."""
    edge_index, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)
    closest_index = torch.zeros((3, 3), dtype=torch.long)
    monkeypatch.setattr(
        "spine.model.grappa.encode.geometric.cdist_fast",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected distance")),
    )

    features = ClustGeoEdgeEncoder(use_numpy=False)(
        graph_data.to_tensor(),
        graph_clusters,
        edge_index,
        closest_index=closest_index,
    )

    assert features.shape == (1, 19)


def test_torch_geometric_edge_encoder_reports_distance_failure(
    graph_data,
    graph_clusters,
    monkeypatch,
):
    """A missing pairwise-distance result raises a focused runtime error."""
    edge_index, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)
    monkeypatch.setattr(
        "spine.model.grappa.encode.geometric.cdist_fast",
        lambda *args: None,
    )

    with pytest.raises(RuntimeError, match="inter-cluster distances"):
        ClustGeoEdgeEncoder(use_numpy=False)(
            graph_data.to_tensor(),
            graph_clusters,
            edge_index,
        )
