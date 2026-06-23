"""Tests for network visualization helpers."""

import numpy as np
import plotly.graph_objs as go
import pytest

from spine.vis.drawer import network as network_module
from spine.vis.drawer.network import network_schematic, network_topology

POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)
CLUSTS = [
    np.array([0, 1, 2], dtype=np.int64),
    np.array([3, 4], dtype=np.int64),
]


def test_network_helpers_build_topology_and_schematic():
    edges = np.array([[0, 1]], dtype=np.int64)

    topology = network_topology(POINTS, CLUSTS, edges, mode="scatter")
    topology_extra_cols = network_topology(
        np.hstack([np.zeros((len(POINTS), 1)), POINTS]),
        CLUSTS,
        edges,
        mode="scatter",
    )
    topology_with_labels = network_topology(
        POINTS,
        CLUSTS,
        edges,
        mode="scatter",
        edge_labels=np.array([1]),
    )
    schematic = network_schematic(CLUSTS, edges, np.array([0, 1]), edge_labels=[1])

    assert len(topology) == 2
    assert len(topology_extra_cols) == 2
    assert topology_with_labels[-1].line.color.tolist() == [1, 1, 1]
    assert len(topology[-1].x) == 3
    assert len(schematic) == 2
    assert schematic[0].type == "scatter"
    assert "Edge label" in schematic[1].text[0]


def test_network_topology_covers_centroid_and_cone_edges():
    edges = np.array([[0, 1]], dtype=np.int64)

    circles = network_topology(POINTS, CLUSTS, edges, mode="circle")
    hulls = network_topology(POINTS, CLUSTS, edges, mode="hull")
    cones = network_topology(POINTS, CLUSTS, edges, mode="cone")

    assert len(circles[-1].x) == 3
    assert len(hulls[-1].x) == 3
    assert len(cones[-1].x) == 3


def test_network_validation_rejects_ambiguous_color_and_labels():
    edges = np.array([[0, 1]], dtype=np.int64)
    labels = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="clust_labels"):
        network_topology(POINTS, CLUSTS, edges, color="red")
    with pytest.raises(ValueError, match="clust_labels"):
        network_schematic(CLUSTS, edges, labels, color="red")
    with pytest.raises(ValueError, match="primary label"):
        network_schematic(CLUSTS, edges, np.array([0]))
    with pytest.raises(ValueError, match="0 or 1"):
        network_schematic(CLUSTS, edges, np.array([0, 2]))


def test_network_topology_rejects_cone_traces_without_coordinates(monkeypatch):
    """Cone edge construction should reject malformed node traces."""
    edges = np.array([[0, 1]], dtype=np.int64)

    monkeypatch.setattr(
        network_module,
        "scatter_clusters",
        lambda *args, **kwargs: [go.Cone(x=None, y=[0.0], z=[0.0])],
    )

    with pytest.raises(ValueError, match="missing coordinate information"):
        network_topology(POINTS, CLUSTS, edges, mode="cone")
