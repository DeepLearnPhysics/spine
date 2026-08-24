"""Tests for graph helpers."""

import numpy as np
import pytest

from spine.math.cluster import dbscan
from spine.math.distance import METRICS
from spine.math.graph import (
    _dfs,
    _dfs_iterative,
    _radius_graph_brute_force,
    bipartite_radius_graph,
    connected_components,
    csr_graph,
    grouped_radius_graph,
    radius_graph,
    shortest_path,
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
    for search in (_dfs, _dfs_iterative):
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


def test_shortest_path_supports_weighted_multi_source_graphs():
    """Dijkstra traversal should preserve distances and closest sources."""
    edges = np.array([[0, 1], [0, 2], [2, 1], [1, 3]], dtype=np.int64)
    weights = np.array([5.0, 1.0, 1.0, 1.0])

    distances, sources = shortest_path(
        edges,
        weights,
        4,
        (
            np.array([0], dtype=np.int64),
            np.array([0.0]),
            np.array([10], dtype=np.int64),
        ),
    )

    # Node one is first queued through the longer direct edge, then relaxed
    # through node two. This also exercises stale priority-queue entries.
    np.testing.assert_allclose(distances, [0.0, 2.0, 1.0, 3.0])
    np.testing.assert_array_equal(sources, [10, 10, 10, 10])

    distances, sources = shortest_path(
        edges[:1],
        weights[:1],
        3,
        (
            np.array([0, 2], dtype=np.int64),
            np.array([2.0, 0.0]),
            np.array([10, 20], dtype=np.int64),
        ),
        directed=False,
    )
    np.testing.assert_allclose(distances, [2.0, 7.0, 0.0])
    np.testing.assert_array_equal(sources, [10, 10, 20])


def test_shortest_path_validates_edge_and_source_shapes():
    """Shortest-path inputs should provide one value per edge and source."""
    edges = np.array([[0, 1]], dtype=np.int64)
    sources = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="graph edge"):
        shortest_path(
            edges,
            np.empty(0),
            2,
            (sources, np.array([0.0]), sources),
        )

    with pytest.raises(ValueError, match="source node"):
        shortest_path(edges, np.ones(1), 2, (sources, np.empty(0), sources))

    with pytest.raises(ValueError, match="identifier"):
        shortest_path(edges, np.ones(1), 2, (sources, np.array([0.0]), np.empty(0)))


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
        radius_graph(points, 1.0, metric_id=99)
    with pytest.raises(ValueError, match="non-negative"):
        radius_graph(points, -1.0)


def test_radius_graph_matches_brute_force_oracle():
    """Spatial hashing should preserve exact brute-force radius edges."""
    rng = np.random.default_rng(17)
    points = rng.uniform(-4.0, 4.0, size=(60, 3)).astype(np.float32)
    points[5] = points[4]

    cases = (
        (METRICS["minkowski"], 1.4, 3.0),
        (METRICS["cityblock"], 1.4, 2.0),
        (METRICS["euclidean"], 1.4, 2.0),
        (METRICS["sqeuclidean"], 1.4**2, 2.0),
        (METRICS["chebyshev"], 1.4, 2.0),
    )
    for metric, radius, p in cases:
        expected = _radius_graph_brute_force(points, radius, metric, p)
        result = radius_graph(points, radius, metric, p, use_hash=True)
        assert sorted_edges(result) == sorted_edges(expected)

    with pytest.raises(ValueError, match="Distance metric"):
        _radius_graph_brute_force(points, 1.0, 99)


def test_grouped_radius_graph_matches_independent_graphs():
    """Grouped backends should match independent per-group constructions."""
    rng = np.random.default_rng(81)
    points = rng.uniform(-3.0, 3.0, size=(80, 3)).astype(np.float32)
    groups = rng.integers(0, 7, size=len(points), dtype=np.int64)

    cases = (
        (METRICS["minkowski"], 1.3, 3.0),
        (METRICS["cityblock"], 1.3, 2.0),
        (METRICS["euclidean"], 1.3, 2.0),
        (METRICS["sqeuclidean"], 1.3**2, 2.0),
        (METRICS["chebyshev"], 1.3, 2.0),
    )
    for metric, radius, p in cases:
        expected = []
        for group in np.unique(groups):
            indexes = np.where(groups == group)[0]
            local_edges = radius_graph(points[indexes], radius, metric, p)
            expected.extend(map(tuple, indexes[local_edges]))

        result = grouped_radius_graph(points, groups, radius, metric, p)
        assert sorted_edges(result) == sorted(expected)


def test_grouped_radius_graph_validates_groups():
    """Grouped graphs should require exactly one scalar group per point."""
    points = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="one identifier"):
        grouped_radius_graph(points, np.zeros(2, dtype=np.int64), 1.0)
    with pytest.raises(ValueError, match="one identifier"):
        grouped_radius_graph(points, np.zeros((3, 1), dtype=np.int64), 1.0)


def test_grouped_components_match_independent_dbscan_calls():
    """One grouped component pass should match DBSCAN run on every group."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    groups = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

    # Reproduce the existing path and make its local component IDs global.
    expected = np.empty(len(points), dtype=np.int64)
    offset = 0
    for group in np.unique(groups):
        indexes = np.where(groups == group)[0]
        labels = dbscan(
            points[indexes],
            eps=1.1,
            metric_id=METRICS["euclidean"],
        )
        expected[indexes] = labels + offset
        offset += np.max(labels) + 1

    edges = grouped_radius_graph(points, groups, 1.1)
    result = connected_components(edges, len(points), directed=False)

    # Component numbering is arbitrary; compare the induced partitions.
    expected_pairs = expected[:, None] == expected[None, :]
    result_pairs = result[:, None] == result[None, :]
    np.testing.assert_array_equal(result_pairs, expected_pairs)


def test_bipartite_radius_graph_matches_combined_oracle():
    """Bipartite cell queries should emit only cross-set radius edges."""
    first = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
        dtype=np.float32,
    )
    second = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [8.0, 8.0, 8.0]],
        dtype=np.float32,
    )

    combined = np.vstack((first, second))
    expected = _radius_graph_brute_force(
        combined,
        np.sqrt(2.0),
        METRICS["euclidean"],
    )
    cross_set = (expected[:, 0] < len(first)) & (expected[:, 1] >= len(first))
    expected = expected[cross_set]
    expected[:, 1] -= len(first)

    result = bipartite_radius_graph(
        first,
        second,
        np.sqrt(2.0),
        METRICS["euclidean"],
    )
    assert sorted_edges(result) == sorted_edges(expected)

    empty = bipartite_radius_graph(first[:0], second, 1.0)
    assert empty.shape == (0, 2)
    empty = bipartite_radius_graph(first, second[:0], 1.0)
    assert empty.shape == (0, 2)


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
