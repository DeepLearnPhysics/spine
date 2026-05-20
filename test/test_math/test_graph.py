"""Tests for graph helpers."""

import numpy as np
import pytest

from spine.math.distance import METRICS
from spine.math.graph import (
    connected_components,
    csr_graph,
    dfs,
    dfs_iterative,
    radius_graph,
    union_find,
)


def sorted_edges(edge_index):
    """Return lexicographically sorted edge tuples."""
    return sorted(map(tuple, np.asarray(edge_index)))


def test_csr_graph_directed_and_undirected_neighbors():
    """CSR graph should expose directed and undirected neighborhoods."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    directed = csr_graph(edges, 3, directed=True)
    np.testing.assert_array_equal(directed[0], [1])
    np.testing.assert_array_equal(directed[2], [])
    assert directed.num_neighbors(1) == 1

    undirected = csr_graph(edges, 3, directed=False)
    np.testing.assert_array_equal(np.sort(undirected[1]), [0, 2])
    assert undirected.num_neighbors(1) == 2


def test_connected_components_and_dfs_variants():
    """Connected-component helpers should traverse equivalent components."""
    edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int64)

    labels = connected_components(edges, 6, directed=False)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 2])

    graph = csr_graph(edges, 6, directed=False)
    for search in (dfs, dfs_iterative):
        visited = np.zeros(6, dtype=np.bool_)
        component = np.empty(6, dtype=np.int64)
        comp_idx = np.zeros(1, dtype=np.int64)
        search(graph, visited, 0, component, comp_idx)
        np.testing.assert_array_equal(np.sort(component[: comp_idx[0]]), [0, 1, 2])


def test_connected_components_respects_min_samples():
    """Nodes below the neighbor threshold should not expand components."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    labels = connected_components(edges, 3, min_samples=4, directed=False)

    np.testing.assert_array_equal(labels, [0, 1, 2])


def test_radius_graph_supports_all_metrics():
    """Radius graph should dispatch all supported distance metrics."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    for metric in (
        METRICS["minkowski"],
        METRICS["cityblock"],
        METRICS["euclidean"],
        METRICS["sqeuclidean"],
        METRICS["chebyshev"],
    ):
        edges = radius_graph(points, 1.1, metric_id=metric, p=2.0)
        assert sorted_edges(edges) == [(0, 1)]

    with pytest.raises(ValueError, match="Distance metric"):
        radius_graph(points, 1.0, metric_id=np.int64(99))


def test_union_find_returns_labels_and_groups():
    """Union-find should merge connected nodes and optionally keep raw labels."""
    edges = np.array([[0, 1], [2, 3]], dtype=np.int64)

    labels, groups = union_find(edges, 5)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(np.sort(groups[0]), [0, 1])
    np.testing.assert_array_equal(np.sort(groups[1]), [2, 3])
    np.testing.assert_array_equal(np.sort(groups[2]), [4])

    raw_labels, _ = union_find(edges, 5, return_inverse=False)
    np.testing.assert_array_equal(raw_labels, [0, 0, 2, 2, 4])


def test_union_find_group_keys_match_returned_labels():
    """Group dictionary keys should use the same label space as labels."""
    edges = np.array([[1, 2], [0, 1]], dtype=np.int64)

    labels, groups = union_find(edges, 5)

    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 2])
    assert set(groups.keys()) == set(labels)
    np.testing.assert_array_equal(np.sort(groups[0]), [0, 1, 2])
    np.testing.assert_array_equal(groups[1], [3])
    np.testing.assert_array_equal(groups[2], [4])


def test_union_find_merges_into_lower_root():
    """Union-find should use a stable low-root representative."""
    edges = np.array([[2, 1]], dtype=np.int64)

    labels, groups = union_find(edges, 3, return_inverse=False)

    np.testing.assert_array_equal(labels, [0, 1, 1])
    np.testing.assert_array_equal(groups[1], [1, 2])


def test_union_find_handles_empty_graph():
    """Union-find should handle a graph with no nodes."""
    labels, groups = union_find(np.empty((0, 2), dtype=np.int64), 0)

    np.testing.assert_array_equal(labels, [])
    assert len(groups) == 0
