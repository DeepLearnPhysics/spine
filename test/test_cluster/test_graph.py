"""Tests for GNN network utility functions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spine.cluster.graph import (
    _get_cluster_edge_features_vec,
    get_cluster_edge_features,
    get_cluster_edge_features_batch,
    get_edge_distances,
    inter_cluster_distance,
)
from spine.cluster.topology import (
    complete_graph,
    filter_invalid_nodes,
    get_fragment_edges,
)
from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch

GRAPH_BASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "spine"
    / "model"
    / "grappa"
    / "graph"
    / "base.py"
)


def load_graph_base():
    """Load GraphBase without importing the full GNN package."""
    spec = importlib.util.spec_from_file_location("graph_base", GRAPH_BASE_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphBase


def test_recursive_distance_algorithm_warns_as_legacy_alias():
    """The recursive distance algorithm name should be visibly legacy."""
    graph_base = load_graph_base()

    class TestGraph(graph_base):
        name = "test"

    # The warning lives in GraphBase, not a specific concrete graph type.
    with pytest.warns(FutureWarning, match="does not perform recursive search"):
        graph = TestGraph(dist_algorithm="recursive")

    assert graph.dist_iterative is True
    assert graph.dist_legacy is True


def test_inter_cluster_distance_can_use_legacy_iterative_closest_pair():
    """Legacy distance mode should preserve historical iterative CPA output."""
    x1 = np.array(
        [
            [0.58366364, -1.8748202, 0.9472971],
            [-0.24740814, 0.6954392, 1.1409228],
            [0.22428122, -0.5900606, 1.20232],
        ],
        dtype=np.float32,
    )
    x2 = np.array(
        [
            [1.3192177, 0.69287896, 1.1638298],
            [-0.6025194, -0.69706947, 2.202115],
            [0.1937491, 0.1192039, 1.1976705],
            [0.3246087, -0.36247766, 1.2971592],
        ],
        dtype=np.float32,
    )
    voxels = np.vstack([x1, x2])
    clusts = [
        np.arange(len(x1), dtype=np.int64),
        np.arange(len(x1), len(x1) + len(x2), dtype=np.int64),
    ]
    counts = np.array([len(clusts)], dtype=np.int64)

    fixed_dist, fixed_index = inter_cluster_distance(
        voxels,
        clusts,
        counts,
        iterative=True,
        return_index=True,
        use_legacy_distance=False,
    )
    legacy_dist, legacy_index = inter_cluster_distance(
        voxels,
        clusts,
        counts,
        iterative=True,
        return_index=True,
        use_legacy_distance=True,
    )

    assert np.isclose(fixed_dist[0, 1], 0.2661843)
    assert fixed_index[0, 1] == 11
    assert np.isclose(legacy_dist[0, 1], 0.7099366)
    assert legacy_index[0, 1] == 10


def test_cluster_edge_features_accept_indexed_float32_coordinates_legacy():
    """Indexed cluster coordinate views should compile in legacy CPA mode."""
    data = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 4.0, 0.0, 4.0],
            [0.0, 4.0, 1.0, 5.0],
            [0.0, 5.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    clusts = [
        np.array([0, 2, 1], dtype=np.int64),
        np.array([3, 5, 4], dtype=np.int64),
    ]
    edge_index = np.array([[0, 1]], dtype=np.int64)

    feats = get_cluster_edge_features(
        data,
        clusts,
        edge_index,
        iterative=False,
        use_legacy_distance=True,
    )

    assert feats.shape == (1, 19)
    assert feats.dtype == np.float32


def test_graph_feature_distance_and_topology_helpers():
    """Graph helpers should cover edge geometry, CPAs, fragments, and filtering."""
    voxels = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [3, 1, 0]], dtype=np.float32)
    clusts = [np.array([0, 1]), np.array([2, 3])]
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)

    features = get_cluster_edge_features(voxels, clusts, edges)
    assert features.shape == (2, 19)
    closest = np.array([[0, 0], [0, 0]], dtype=np.int64)
    assert get_cluster_edge_features(voxels, clusts, edges, closest).shape == (2, 19)
    assert get_cluster_edge_features(
        voxels, [], np.empty((0, 2), dtype=np.int64)
    ).shape == (0, 19)
    assert _get_cluster_edge_features_vec(voxels, clusts, edges.T).shape == (2, 19)
    assert _get_cluster_edge_features_vec(voxels, clusts, edges.T, closest).shape == (
        2,
        19,
    )

    lengths, left, right = get_edge_distances(voxels, clusts, edges.T, False)
    assert lengths.shape == left.shape == right.shape == (2,)
    loop_lengths, _, _ = get_edge_distances(
        voxels, clusts, np.array([[0], [0]], dtype=np.int64), False
    )
    assert loop_lengths[0] == 0
    assert get_edge_distances(voxels, clusts, edges.T, False, True)[0].shape == (2,)

    assert inter_cluster_distance(voxels, clusts).shape == (2, 2)
    assert inter_cluster_distance(voxels, clusts, centroid=True).shape == (2, 2)
    assert inter_cluster_distance(
        voxels, clusts, iterative=True, use_legacy_distance=True
    ).shape == (2, 2)
    assert inter_cluster_distance(voxels, [], return_index=False).shape == (0, 0)
    distance, index = inter_cluster_distance(voxels, [], return_index=True)
    assert distance.shape == index.shape == (0, 0)
    with pytest.raises(AssertionError, match="centroid"):
        inter_cluster_distance(voxels, clusts, centroid=True, return_index=True)

    np.testing.assert_array_equal(
        complete_graph(np.array([2, 1])), np.array([[0], [1]])
    )
    np.testing.assert_array_equal(
        get_fragment_edges(np.array([[10, 20], [20, 99]]), np.array([10, 20])),
        [[0, 1]],
    )

    data = TensorBatch(voxels, counts=[4], coord_cols=np.arange(3))
    clusters = IndexBatch(clusts, spans=[4], counts=[2], single_counts=[2, 2])
    directed = EdgeIndexBatch(edges.T, counts=[2], spans=[2], directed=True)
    assert get_cluster_edge_features_batch(data, clusters, directed).shape == (2, 19)
    undirected_edges = np.array([[0, 1], [1, 0]], dtype=np.int64).T
    undirected = EdgeIndexBatch(undirected_edges, counts=[2], spans=[2], directed=False)


def test_filter_invalid_nodes_bridges_tree_edges():
    """Invalid roots, internal nodes, leaves, and invalid parentage are handled."""
    edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
    np.testing.assert_array_equal(
        filter_invalid_nodes(edges, np.array([2])), [[0, 1], [1, 3]]
    )
    np.testing.assert_array_equal(
        filter_invalid_nodes(edges, np.array([1])), [[0, 2], [0, 3]]
    )
    np.testing.assert_array_equal(
        filter_invalid_nodes(edges, np.array([0])), [[1, 2], [1, 3]]
    )
    with pytest.raises(AssertionError, match="multiple parents"):
        filter_invalid_nodes(
            np.array([[0, 2], [1, 2], [2, 3]], dtype=np.int64), np.array([2])
        )
