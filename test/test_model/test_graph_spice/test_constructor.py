"""Direct tests for Graph-SPICE graph construction and evaluation."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from spine.data import TensorBatch
from spine.model.graph_spice.constructor import ClusterGraphConstructor
from spine.model.graph_spice.orphan import OrphanAssigner


def constructor_config(name="knn"):
    """Return a minimal two-shape graph constructor configuration."""
    graph = {"name": name, "k": 1} if name == "knn" else {"name": name, "r": 2.0}
    return {
        "graph": graph,
        "shapes": ["shower", "track"],
        "edge_threshold": 0.5,
        "kernel_fn": lambda left, right: -torch.sum((left - right) ** 2, dim=1),
        "label_edges": True,
    }


def graph_inputs():
    """Build one event containing one populated and one absent shape."""
    coords = TensorBatch(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        counts=[3],
    )
    features = TensorBatch(torch.tensor([[0.0], [0.1], [0.2]]), counts=[3])
    shapes = TensorBatch(torch.zeros(3, dtype=torch.long), counts=[3])
    labels = TensorBatch(torch.tensor([0, 0, 1]), counts=[3])
    return coords, features, shapes, labels


def test_constructor_validates_and_builds_labeled_graphs():
    """Supported graph modes should build edges and preserve empty shapes."""
    for name in ("knn", "radius"):
        constructor = ClusterGraphConstructor(**deepcopy(constructor_config(name)))
        graph = constructor(*graph_inputs())
        assert graph["edge_index"].shape[1] == 2
        assert graph["edge_label"].shape == graph["edge_prob"].shape
        assert graph["node_clusts"].counts.tolist() == [2]
        assert graph["node_clusts"].single_counts[-1] == 0

    with pytest.raises(AssertionError, match="graph constructor"):
        ClusterGraphConstructor({}, ["shower"], 0.5)
    with pytest.raises(ValueError, match="not recognized"):
        ClusterGraphConstructor({"name": "bad"}, ["shower"], 0.5)


def test_constructor_prediction_evaluation_and_entry_selection():
    """Prediction, metrics, and graph slicing should cover populated/empty shapes."""
    constructor = ClusterGraphConstructor(**deepcopy(constructor_config()))
    graph = constructor(*graph_inputs())
    clusts, clust_shapes = constructor.fit_predict(graph)
    assert clusts.counts.tolist() == [1]
    assert clust_shapes.counts.tolist() == [1]

    graph["node_label"] = TensorBatch(torch.tensor([0, 0, 1]), counts=[3])
    metrics = constructor.evaluate(graph)
    assert len(metrics["ari"]) == 1
    assert metrics["ari_1"] == [1.0]
    averaged = constructor.evaluate(graph, mean=True)
    assert np.isscalar(averaged["purity"])

    entry = constructor.get_entry(graph, 0, 0)
    assert len(entry["node_coords"]) == 3
    assert len(entry["edge_index"]) == len(graph["edge_clusts"][0][0])
    graph["unexpected"] = object()
    with pytest.raises(KeyError, match="not recognized"):
        constructor.get_entry(graph, 0, 0)

    constructor.invert = False
    assert constructor.fit_predict(graph, threshold=0.0)[0].counts.tolist() == [1]


def test_orphan_assigner_mode_validation_and_radius_setup():
    """Orphan assignment should validate modes and construct radius fallback."""
    with pytest.raises(ValueError, match="knn.*radius"):
        OrphanAssigner("bad")
    assigner = OrphanAssigner("radius", radius=1.0, assign_all=True)
    assert assigner.dbscan.eps == 1.0
    assert OrphanAssigner("knn", k=1).mode == "knn"
