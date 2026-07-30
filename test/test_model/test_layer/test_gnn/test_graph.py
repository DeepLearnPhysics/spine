"""Behavioral tests for GNN graph constructors."""

import numpy as np
import pytest

from spine.model.layer.gnn.graph import (
    BipartiteGraph,
    CompleteGraph,
    DelaunayGraph,
    KNNGraph,
    LoopGraph,
    MSTGraph,
)


@pytest.mark.parametrize(
    ("constructor", "expected_counts"),
    [
        (CompleteGraph(directed=True), [1, 0]),
        (KNNGraph(k=1, directed=True), [2, 0]),
        (MSTGraph(directed=True), [1, 0]),
        (DelaunayGraph(directed=True), [1, 0]),
        (BipartiteGraph(directed=True), [1, 0]),
        (LoopGraph(directed=True), [2, 1]),
    ],
)
def test_graph_constructor_returns_consistent_edge_counts(
    graph_data,
    graph_clusters,
    constructor,
    expected_counts,
):
    edge_index, _, _ = constructor(graph_data, graph_clusters)

    assert edge_index.index.shape[0] == 2
    assert edge_index.index.shape[1] == sum(expected_counts)
    assert np.array_equal(edge_index.counts, expected_counts)


def test_undirected_graph_interleaves_reciprocal_edges(
    graph_data,
    graph_clusters,
):
    edge_index, _, _ = CompleteGraph(directed=False)(
        graph_data,
        graph_clusters,
    )

    assert np.array_equal(edge_index.counts, [2, 0])
    assert np.array_equal(edge_index.index[:, 0], edge_index.index[::-1, 1])


def test_knn_rejects_nonpositive_neighbor_count():
    with pytest.raises(ValueError, match="positive"):
        KNNGraph(k=0)


def test_undirected_knn_deduplicates_neighbor_pairs(
    graph_data,
    graph_clusters,
):
    edge_index, _, _ = KNNGraph(k=1, directed=False)(
        graph_data,
        graph_clusters,
    )

    assert np.array_equal(edge_index.counts, [2, 0])


def test_mst_connects_clusters_at_zero_distance():
    edge_index = MSTGraph._generate(
        np.array([0, 0], dtype=np.int64),
        np.zeros((2, 2), dtype=np.float64),
    )

    assert edge_index.shape == (2, 1)
