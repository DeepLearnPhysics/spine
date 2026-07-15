from __future__ import annotations

import numpy as np
import pytest

import spine.ana.diag.graph as graph_mod
from spine.ana.diag.graph import GraphEdgeLengthAna


def test_graph_edge_length_ana_accepts_tuple_object_types(monkeypatch):
    writers = []
    monkeypatch.setattr(
        GraphEdgeLengthAna, "initialize_writer", lambda self, name: writers.append(name)
    )

    ana = GraphEdgeLengthAna(obj_type=("particle", "interaction"))

    assert ana.obj_type == ["particle", "interaction", "fragment"]
    assert set(writers) == {
        "truth_particles",
        "truth_interactions",
    }


def test_graph_edge_length_ana_accepts_none_and_reco_modes(monkeypatch):
    writers = []
    monkeypatch.setattr(
        GraphEdgeLengthAna, "initialize_writer", lambda self, name: writers.append(name)
    )

    empty = GraphEdgeLengthAna(obj_type=None)
    reco = GraphEdgeLengthAna(obj_type="interaction", run_mode="reco")

    assert empty.obj_type == []
    assert reco.obj_type == ["interaction", "particle"]
    assert "points" in reco.keys
    assert writers == ["reco_particles", "reco_interactions"]


def test_graph_edge_length_ana_validates_time_window(monkeypatch):
    monkeypatch.setattr(
        GraphEdgeLengthAna, "initialize_writer", lambda self, name: None
    )

    with pytest.raises(ValueError, match="two values"):
        GraphEdgeLengthAna(time_window=(0.0, 1.0, 2.0))

    with pytest.raises(ValueError, match="Cannot restrict timing"):
        GraphEdgeLengthAna(time_window=(0.0, 1.0), run_mode="reco")


class FakeConstituent:
    def __init__(self, id, index, shape):
        self.id = id
        self.index = index
        self.shape = shape


class FakeGroup:
    id = 10
    is_truth = True
    t = 5.0

    def __init__(self, constituents):
        self.fragments = constituents


def test_graph_edge_length_ana_process_stores_distances(monkeypatch):
    rows = []
    monkeypatch.setattr(
        GraphEdgeLengthAna, "initialize_writer", lambda self, name: None
    )
    monkeypatch.setattr(
        GraphEdgeLengthAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    monkeypatch.setattr(
        graph_mod,
        "inter_cluster_distance",
        lambda points, indexes: np.array([[0.0, 3.0], [4.0, 0.0]]),
    )
    constituents = [
        FakeConstituent(1, np.array([0]), 0),
        FakeConstituent(2, np.array([1]), 1),
    ]
    ana = GraphEdgeLengthAna(obj_type="particle", time_window=(0.0, 10.0))

    ana.process(
        {
            "truth_particles": [FakeGroup(constituents)],
            "points_label": np.zeros((2, 3), dtype=np.float32),
        }
    )

    assert rows == [
        (
            "truth_particles",
            {
                "id": 10,
                "source_id": 1,
                "sink_id": 2,
                "source_shape": 0,
                "sink_shape": 1,
                "length": 3.0,
            },
        )
    ]


def test_graph_edge_length_ana_processes_particle_constituents(monkeypatch):
    rows = []
    monkeypatch.setattr(
        GraphEdgeLengthAna, "initialize_writer", lambda self, name: None
    )
    monkeypatch.setattr(
        GraphEdgeLengthAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    monkeypatch.setattr(
        graph_mod,
        "inter_cluster_distance",
        lambda points, indexes: np.array([[0.0, 2.0], [2.0, 0.0]]),
    )

    class InteractionGroup(FakeGroup):
        def __init__(self, constituents):
            self.particles = constituents

    constituents = [
        FakeConstituent(1, np.array([0]), 0),
        FakeConstituent(2, np.array([1]), 1),
    ]
    ana = GraphEdgeLengthAna(obj_type="interaction")

    ana.process(
        {
            "truth_particles": [],
            "truth_interactions": [InteractionGroup(constituents)],
            "points_label": np.zeros((2, 3), dtype=np.float32),
        }
    )

    assert rows[0][1]["length"] == 2.0


def test_graph_edge_length_ana_pairs_shape_indexes_with_offset(monkeypatch):
    rows = []
    monkeypatch.setattr(
        GraphEdgeLengthAna, "initialize_writer", lambda self, name: None
    )
    monkeypatch.setattr(
        GraphEdgeLengthAna,
        "append",
        lambda self, name, **kwargs: rows.append(kwargs),
    )
    monkeypatch.setattr(
        graph_mod,
        "inter_cluster_distance",
        lambda points, indexes: np.array(
            [
                [0.0, 9.0, 8.0, 7.0],
                [9.0, 0.0, 6.0, 5.0],
                [8.0, 6.0, 0.0, 4.0],
                [7.0, 5.0, 4.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    constituents = [
        FakeConstituent(1, np.array([0]), 0),
        FakeConstituent(2, np.array([1]), 1),
        FakeConstituent(3, np.array([2]), 1),
        FakeConstituent(4, np.array([3]), 2),
    ]
    ana = GraphEdgeLengthAna(obj_type="particle")
    group = FakeGroup(constituents)

    ana.store_distances(
        "truth_particles", np.zeros((4, 3), dtype=np.float32), group, constituents
    )

    shape_pairs = {(row["source_shape"], row["sink_shape"]) for row in rows}
    assert (1, 2) in shape_pairs


def test_graph_edge_length_ana_skips_out_of_time_groups(monkeypatch):
    rows = []
    monkeypatch.setattr(
        GraphEdgeLengthAna, "initialize_writer", lambda self, name: None
    )
    monkeypatch.setattr(
        GraphEdgeLengthAna,
        "append",
        lambda self, name, **kwargs: rows.append(kwargs),
    )
    group = FakeGroup(
        [FakeConstituent(1, np.array([0]), 0), FakeConstituent(2, np.array([1]), 1)]
    )
    group.t = 20.0
    ana = GraphEdgeLengthAna(obj_type="particle", time_window=(0.0, 10.0))

    ana.store_distances("truth_particles", np.zeros((2, 3)), group, group.fragments)

    assert rows == []


def test_graph_edge_length_ana_skips_single_constituent(monkeypatch):
    rows = []
    monkeypatch.setattr(
        GraphEdgeLengthAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    ana = GraphEdgeLengthAna(obj_type="particle")
    group = FakeGroup([FakeConstituent(1, np.array([0]), 0)])

    ana.store_distances("truth_particles", np.zeros((1, 3)), group, group.fragments)

    assert rows == []
