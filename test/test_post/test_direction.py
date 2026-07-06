from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import spine.post.reco.direction as direction_mod
from spine.constants import TRACK_SHP
from spine.post.reco.direction import DirectionProcessor


def test_direction_processor_assigns_track_start_and_end_dirs(monkeypatch):
    monkeypatch.setattr(
        direction_mod,
        "get_cluster_directions",
        lambda points, ref_points, clusts, **kwargs: np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
    )
    particle = SimpleNamespace(
        id=0,
        is_truth=False,
        shape=TRACK_SHP,
        index=np.asarray([0, 1], dtype=np.int32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        end_point=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        start_dir=np.zeros(3, dtype=np.float32),
        end_dir=np.zeros(3, dtype=np.float32),
    )
    processor = DirectionProcessor(obj_type="particle", run_mode="reco")

    processor.process(
        {
            "reco_particles": [particle],
            "points": np.zeros((2, 3), dtype=np.float32),
        }
    )

    assert np.allclose(particle.start_dir, [1.0, 0.0, 0.0])
    assert np.allclose(particle.end_dir, [-0.0, -1.0, -0.0])


def test_direction_processor_skips_empty_objects(monkeypatch):
    calls = []
    monkeypatch.setattr(
        direction_mod,
        "get_cluster_directions",
        lambda *args, **kwargs: calls.append(args),
    )
    particle = SimpleNamespace(
        id=0,
        is_truth=False,
        shape=TRACK_SHP,
        index=np.empty(0, dtype=np.int32),
        start_point=np.zeros(3, dtype=np.float32),
        end_point=np.ones(3, dtype=np.float32),
    )
    processor = DirectionProcessor(obj_type="particle", run_mode="reco")

    processor.process(
        {
            "reco_particles": [particle],
            "points": np.zeros((0, 3), dtype=np.float32),
        }
    )

    assert calls == []


def test_direction_processor_assigns_truth_reco_dirs(monkeypatch):
    monkeypatch.setattr(
        direction_mod,
        "get_cluster_directions",
        lambda points, ref_points, clusts, **kwargs: np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
    )
    particle = SimpleNamespace(
        id=0,
        is_truth=True,
        shape=TRACK_SHP,
        index=np.asarray([0, 1], dtype=np.int32),
        points=np.zeros((2, 3), dtype=np.float32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        end_point=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        reco_start_dir=np.zeros(3, dtype=np.float32),
        reco_end_dir=np.zeros(3, dtype=np.float32),
    )
    processor = DirectionProcessor(obj_type="particle", run_mode="truth")

    processor.process(
        {
            "truth_particles": [particle],
            "points_label": np.zeros((2, 3), dtype=np.float32),
        }
    )

    assert np.allclose(particle.reco_start_dir, [-1.0, -0.0, -0.0])
    assert np.allclose(particle.reco_end_dir, [-0.0, -1.0, -0.0])


def test_direction_processor_skips_shower_end_dir(monkeypatch):
    monkeypatch.setattr(
        direction_mod,
        "get_cluster_directions",
        lambda points, ref_points, clusts, **kwargs: np.asarray(
            [[1.0, 0.0, 0.0]], dtype=np.float32
        ),
    )
    particle = SimpleNamespace(
        id=0,
        is_truth=False,
        shape=0,
        index=np.asarray([0, 1], dtype=np.int32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        end_point=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        start_dir=np.zeros(3, dtype=np.float32),
        end_dir=np.zeros(3, dtype=np.float32),
    )
    processor = DirectionProcessor(obj_type="particle", run_mode="reco")

    processor.process(
        {"reco_particles": [particle], "points": np.zeros((2, 3), dtype=np.float32)}
    )

    np.testing.assert_allclose(particle.start_dir, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(particle.end_dir, [0.0, 0.0, 0.0])
