"""Behavioral tests for GNN graph constructors."""

from types import SimpleNamespace

import numpy as np
import pytest

from spine.model.grappa.graph import (
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
    graph_labels,
    graph_clusters,
    constructor,
    expected_counts,
):
    data = graph_labels if isinstance(constructor, BipartiteGraph) else graph_data
    edge_index, _, _ = constructor(data, graph_clusters)

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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dist_method": "bad"}, "method not recognized"),
        ({"dist_algorithm": "bad"}, "algorithm not recognized"),
        ({"max_length": [1.0]}, "provide the list of classes"),
        (
            {"max_length": [1.0, 2.0], "classes": [0, 1]},
            "upper triangular",
        ),
    ],
)
def test_graph_base_validates_distance_configuration(kwargs, message):
    """Distance modes and class-dependent cut matrices are validated early."""
    with pytest.raises(ValueError, match=message):
        CompleteGraph(**kwargs)


def test_graph_base_builds_class_cut_matrix_and_warns_for_legacy_distance():
    """Upper-triangle cuts are symmetric and legacy distance is explicit."""
    graph = CompleteGraph(
        max_length=[1.0, 2.0, 3.0],
        classes=[0, 1],
    )
    np.testing.assert_allclose(graph.max_length, [[1.0, 2.0], [2.0, 3.0]])
    with pytest.warns(FutureWarning, match="legacy"):
        legacy = CompleteGraph(dist_algorithm="recursive")
    assert legacy.dist_iterative and legacy.dist_legacy


def test_graph_filters_groups_class_lengths_and_edge_overflow(
    graph_data,
    graph_clusters,
):
    """Post-construction filters update per-event edge counts consistently."""
    from spine.data import TensorBatch

    edge_index, _, _ = CompleteGraph(directed=True)(
        graph_data,
        graph_clusters,
        groups=TensorBatch(np.array([0, 1, 0]), [2, 1]),
    )
    assert edge_index.counts.tolist() == [0, 0]

    classes = TensorBatch(np.array([0, 1, 0]), [2, 1])
    graph = CompleteGraph(
        directed=True,
        max_length=[10.0, 0.0, 10.0],
        classes=[0, 1],
    )
    edge_index, _, _ = graph(graph_data, graph_clusters, classes=classes)
    assert edge_index.counts.tolist() == [0, 0]

    with pytest.warns(UserWarning, match="too many edges"):
        edge_index, _, _ = CompleteGraph(directed=True, max_count=0)(
            graph_data,
            graph_clusters,
        )
    assert edge_index.counts.tolist() == [0, 0]


def test_graph_restriction_and_required_distance_guards(graph_clusters, graph_data):
    """Direct graph helpers reject unavailable cut and distance context."""
    graph = CompleteGraph()
    with pytest.raises(RuntimeError, match="without a maximum length"):
        graph.restrict(np.empty((2, 0), int), np.zeros(2, int), np.zeros((3, 3)))

    graph.max_length = np.ones((2, 2))
    with pytest.raises(ValueError, match="require cluster classes"):
        graph.restrict(
            np.array([[0], [1]]),
            np.array([1, 0]),
            np.ones((3, 3)),
        )
    with pytest.raises(ValueError, match="requires `dist_mat`"):
        KNNGraph(k=1).generate(data=graph_data, clusts=graph_clusters, dist_mat=None)
    with pytest.raises(ValueError, match="requires `dist_mat`"):
        MSTGraph().generate(data=graph_data, clusts=graph_clusters, dist_mat=None)


def test_loop_and_bipartite_validate_orientation(graph_data, graph_clusters):
    """Special graph topologies enforce their direction and truth contracts."""
    with pytest.raises(ValueError, match="loop graphs"):
        LoopGraph(directed=False)
    with pytest.raises(ValueError, match="directed_to"):
        BipartiteGraph(directed_to="both")
    with pytest.raises(TypeError, match="structured cluster labels"):
        BipartiteGraph(directed=True)(graph_data, graph_clusters)

    edges = BipartiteGraph._generate(
        np.array([0, 0]),
        np.array([True, False]),
        directed=True,
        directed_to="primary",
    )
    assert edges[:, 0].tolist() == [1, 0]
    with pytest.raises(ValueError, match="orientation"):
        BipartiteGraph._generate(
            np.array([0, 0]),
            np.array([True, False]),
            directed=True,
            directed_to="bad",
        )


def test_delaunay_falls_back_for_degenerate_triangulation(monkeypatch):
    """Degenerate voxel geometry falls back to a complete cluster graph."""
    import spine.model.grappa.graph.delaunay as module

    cluster_ids = np.array([0, 1])
    clusters = [np.array([0, 1]), np.array([2, 3])]
    points = np.zeros((4, 3))

    monkeypatch.setattr(
        module,
        "Delaunay",
        lambda *_args, **_kwargs: SimpleNamespace(simplices=np.array([[0, 1]])),
    )
    edges = DelaunayGraph._generate_entry(points, clusters, cluster_ids)
    assert edges.tolist() == [[0], [1]]
