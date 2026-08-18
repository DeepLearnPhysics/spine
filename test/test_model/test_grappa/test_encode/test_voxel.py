"""Tests for restored voxel-level GrapPA node construction and features."""

import numpy as np
import pytest
import torch

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.model.grappa.encode.voxel import (
    VoxelGeoNodeEncoder,
    get_voxel_edge_features,
    get_voxel_edge_features_batch,
    get_voxel_features,
    get_voxel_features_batch,
)
from spine.model.grappa.model import GrapPA


def point_batch() -> TensorBatch:
    """Return two small events with three-dimensional point coordinates."""
    rows = torch.tensor(
        [
            [0, 0.0, 0.0, 0.0, 1.0],
            [0, 1.0, 0.0, 0.0, 1.0],
            [1, 100.0, 0.0, 0.0, 1.0],
            [1, 101.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return TensorBatch(rows, counts=[2, 2], has_batch_col=True, coord_cols=(1, 2, 3))


def test_voxel_features_are_finite_and_batch_local():
    """Neighborhood counts must not leak across event boundaries."""
    features = get_voxel_features_batch(point_batch(), max_dist=2.0)

    assert features.shape == (4, 16)
    assert torch.isfinite(features.torch_tensor()).all()
    assert features.torch_tensor()[:, -1].tolist() == [2.0, 2.0, 2.0, 2.0]

    isolated = get_voxel_features(np.zeros((1, 3), dtype=np.float32))
    assert isolated[0, -1] == 1.0
    assert np.all(isolated[0, 3:15] == 0.0)
    with pytest.raises(ValueError, match="three coordinates"):
        get_voxel_features(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="max_dist"):
        get_voxel_features(np.zeros((1, 3), dtype=np.float32), max_dist=0.0)
    with pytest.raises(ValueError, match="max_dist"):
        get_voxel_features_batch(point_batch(), max_dist=0.0)


def test_voxel_features_support_numpy_and_degenerate_neighborhoods():
    """NumPy batches and rank-deficient neighborhoods should remain stable."""
    rows = point_batch().tensor.numpy()
    data = TensorBatch(rows, counts=[2, 2], has_batch_col=True, coord_cols=(1, 2, 3))
    result = get_voxel_features_batch(data, max_dist=2.0)
    assert result.is_numpy
    assert np.isfinite(result.tensor).all()

    repeated = np.zeros((3, 3), dtype=np.float32)
    assert np.all(get_voxel_features(repeated)[:, 3:15] == 0.0)

    asymmetric = np.array(
        [
            [2.0409191, -2.5556650, 0.4180988],
            [-0.5677696, -0.4526493, -0.2155972],
            [-2.0199862, -0.2319324, -0.8652131],
            [3.3229995, 0.2257866, -0.3526308],
            [-0.2812874, -0.6680464, -1.0551505],
        ],
        dtype=np.float32,
    )
    assert np.isfinite(get_voxel_features(asymmetric, max_dist=10.0)).all()


def test_voxel_edge_features_follow_voxel_graph_directionality():
    """Voxel edge encoding should align feature rows with directed graph edges."""
    coordinates = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    features = get_voxel_edge_features(coordinates, edges)
    assert features.shape == (2, 19)

    data = point_batch()
    directed = EdgeIndexBatch(
        torch.tensor(edges.T), counts=[2, 0], spans=[2, 2], directed=True
    )
    assert get_voxel_edge_features_batch(data, directed).shape == (2, 19)

    undirected = EdgeIndexBatch(
        torch.tensor(edges.T), counts=[2, 0], spans=[2, 2], directed=False
    )
    assert get_voxel_edge_features_batch(data, undirected).shape == (1, 19)


def test_voxel_encoder_selects_singleton_nodes():
    """The encoder should align one local feature row with each voxel node."""
    data = point_batch()
    clusters = [torch.tensor([index]) for index in range(4)]
    clusts = IndexBatch(clusters, data.counts, [2, 2], [1, 1, 1, 1])

    result = VoxelGeoNodeEncoder(max_dist=2.0)(data, clusts)

    assert result.shape == (4, 16)
    assert result.counts.tolist() == [2, 2]
    with pytest.raises(ValueError, match="singleton"):
        bad = IndexBatch([torch.tensor([0, 1])], [2], [1], [2])
        VoxelGeoNodeEncoder()(
            data.select(torch.tensor([True, True, False, False])), bad
        )
    with pytest.raises(ValueError, match="max_dist"):
        VoxelGeoNodeEncoder(max_dist=0.0)


def test_grappa_voxel_source_builds_singleton_clusters():
    """The explicit voxel source should not require structured cluster truth."""
    model = GrapPA.__new__(GrapPA)
    torch.nn.Module.__init__(model)
    model.node_source = "voxel"
    model.node_type = [0, 1, 2, 3]
    model.node_min_size = 1
    model.dbscan = None

    clusts = model._make_clusters(point_batch())

    assert clusts.counts.tolist() == [2, 2]
    assert clusts.single_counts.tolist() == [1, 1, 1, 1]
    assert clusts.full_index.tolist() == [0, 1, 2, 3]
