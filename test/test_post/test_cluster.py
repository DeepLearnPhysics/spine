from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from spine.post.reco.cluster import TrackClusterer


def test_track_clusterer_validates_configuration():
    with pytest.raises(AssertionError):
        TrackClusterer(split_volume="detector")

    with pytest.raises(AssertionError):
        TrackClusterer(particle_type="particle")


def test_track_clusterer_processes_points_without_volume_split():
    processor = TrackClusterer(eps=0.5, min_size=2, max_rel_spread=0.01)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    depositions = np.arange(len(points), dtype=np.float32)

    result = processor.process({"points": points, "depositions": depositions})

    particles = result["reco_particles"]
    assert len(particles) == 2
    assert [part.size for part in particles] == [2, 2]
    assert np.array_equal(particles[0].index, np.array([0, 1]))
    assert np.array_equal(particles[1].index, np.array([2, 3]))


def test_track_clusterer_processes_points_with_volume_split():
    processor = TrackClusterer(
        eps=0.5, min_size=2, max_rel_spread=0.01, split_volume="tpc"
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    depositions = np.arange(len(points), dtype=np.float32)
    sources = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [0, 2]], dtype=np.int64)

    result = processor.process(
        {"points": points, "depositions": depositions, "sources": sources}
    )

    particles = result["reco_particles"]
    assert len(particles) == 2
    assert np.array_equal(particles[0].sources, sources[[0, 1]])
    assert np.array_equal(particles[1].sources, sources[[2, 3]])


def test_track_clusterer_processes_points_with_module_split():
    processor = TrackClusterer(
        eps=0.5, min_size=2, max_rel_spread=0.01, split_volume="module"
    )
    points = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    depositions = np.ones(len(points), dtype=np.float32)
    sources = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)

    result = processor.process(
        {"points": points, "depositions": depositions, "sources": sources}
    )

    assert len(result["reco_particles"]) == 1


def test_track_clusterer_process_volume_quality_cuts():
    processor = TrackClusterer(eps=1.0, min_size=2, max_rel_spread=0.01)

    assert processor.process_volume(np.array([[0.0, 0.0, 0.0]], dtype=np.float32)) == []

    processor = TrackClusterer(
        eps=2.0, min_size=2, max_rel_spread=1.0, max_axis_dist=0.01
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert processor.process_volume(points) == []

    processor = TrackClusterer(eps=0.5, min_size=1, max_rel_spread=1.0)
    processor.dbscan = cast(
        Any,
        SimpleNamespace(fit_predict=lambda points: np.full(len(points), -1)),
    )
    assert processor.process_volume(np.zeros((2, 3), dtype=np.float32)) == []

    processor = TrackClusterer(eps=2.0, min_size=2, max_rel_spread=0.01)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
        dtype=np.float32,
    )
    assert processor.process_volume(points) == []
