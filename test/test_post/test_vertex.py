from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spine.constants import SHOWR_SHP, TRACK_SHP
from spine.post.reco import vertex as vertex_mod


def make_particle(
    shape=TRACK_SHP,
    start_point=(0.0, 0.0, 0.0),
    end_point=(1.0, 0.0, 0.0),
    start_dir=(1.0, 0.0, 0.0),
    is_primary=True,
):
    start_point = np.asarray(start_point, dtype=np.float32)
    end_point = np.asarray(end_point, dtype=np.float32)
    start_dir = np.asarray(start_dir, dtype=np.float32)
    return SimpleNamespace(
        size=2,
        shape=shape,
        is_primary=is_primary,
        start_point=start_point,
        end_point=end_point,
        start_dir=start_dir,
        end_dir=-start_dir,
    )


def test_vertex_processor_reconstructs_reco_vertex_and_updates_tracks(monkeypatch):
    monkeypatch.setattr(
        vertex_mod,
        "get_vertex",
        lambda *args, **kwargs: (np.array([0.0, 0.0, 0.0], dtype=np.float32), "track"),
    )
    track = make_particle(start_point=(5.0, 0.0, 0.0), end_point=(0.0, 0.0, 0.0))
    shower = make_particle(shape=SHOWR_SHP, start_point=(0.5, 0.0, 0.0))
    other = make_particle(shape=99)
    inter = SimpleNamespace(
        is_truth=False,
        particles=[track, shower, other],
        vertex=np.full(3, -1.0, dtype=np.float32),
    )
    processor = vertex_mod.VertexProcessor(
        run_mode="reco", update_orientations=True, update_primaries=True
    )

    processor.process({"reco_interactions": [inter]})

    assert np.array_equal(inter.vertex, np.zeros(3, dtype=np.float32))
    assert np.array_equal(track.start_point, np.zeros(3, dtype=np.float32))
    assert track.is_primary is True
    assert shower.is_primary is True
    assert other.is_primary is False


def test_vertex_processor_reconstructs_truth_vertex(monkeypatch):
    monkeypatch.setattr(
        vertex_mod,
        "get_vertex",
        lambda *args, **kwargs: (np.array([1.0, 2.0, 3.0], dtype=np.float32), "track"),
    )
    inter = SimpleNamespace(
        is_truth=True,
        particles=[make_particle()],
        reco_vertex=np.full(3, -1.0, dtype=np.float32),
    )
    processor = vertex_mod.VertexProcessor(run_mode="truth")

    processor.process({"truth_interactions": [inter]})

    assert np.array_equal(inter.reco_vertex, np.array([1.0, 2.0, 3.0]))


def test_vertex_processor_falls_back_to_nonprimary_particles(monkeypatch):
    monkeypatch.setattr(
        vertex_mod,
        "get_vertex",
        lambda *args, **kwargs: (np.array([2.0, 0.0, 0.0], dtype=np.float32), "track"),
    )
    particle = make_particle(is_primary=False)
    inter = SimpleNamespace(
        is_truth=False,
        particles=[particle],
        vertex=np.full(3, -1.0, dtype=np.float32),
    )
    processor = vertex_mod.VertexProcessor(run_mode="reco")

    processor.reconstruct_vertex_single(inter)

    assert np.array_equal(inter.vertex, np.array([2.0, 0.0, 0.0]))


def test_vertex_processor_skips_empty_interaction():
    inter = SimpleNamespace(
        is_truth=False,
        particles=[SimpleNamespace(size=0, shape=TRACK_SHP)],
        vertex=np.full(3, -1.0, dtype=np.float32),
    )
    processor = vertex_mod.VertexProcessor(run_mode="reco")

    processor.reconstruct_vertex_single(inter)

    assert np.array_equal(inter.vertex, np.full(3, -1.0, dtype=np.float32))


def test_vertex_processor_updates_shower_primary_by_angle(monkeypatch):
    monkeypatch.setattr(
        vertex_mod,
        "get_vertex",
        lambda *args, **kwargs: (np.array([0.0, 0.0, 0.0], dtype=np.float32), "track"),
    )
    shower = make_particle(
        shape=SHOWR_SHP,
        start_point=(2.0, 0.0, 0.0),
        start_dir=(1.0, 0.0, 0.0),
        is_primary=False,
    )
    inter = SimpleNamespace(
        is_truth=False,
        particles=[shower],
        vertex=np.full(3, -1.0, dtype=np.float32),
    )
    processor = vertex_mod.VertexProcessor(
        run_mode="reco",
        update_primaries=True,
        touching_threshold=1.0,
        angle_threshold=0.5,
    )

    processor.reconstruct_vertex_single(inter)

    assert shower.is_primary is True
