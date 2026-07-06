from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import spine.post.reco.points as points_mod
from spine.constants import TRACK_SHP
from spine.post.reco.points import TrackExtremaProcessor


def _track_particle():
    return SimpleNamespace(
        shape=TRACK_SHP,
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        depositions=np.asarray([1.0, 2.0], dtype=np.float32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        end_point=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        start_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        end_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_track_extrema_processor_flips_local_orientation(monkeypatch):
    monkeypatch.setattr(
        points_mod, "check_track_orientation", lambda *args, **kwargs: False
    )
    particle = _track_particle()
    processor = TrackExtremaProcessor(method="local")

    processor.process({"reco_particles": [particle]})

    assert np.allclose(particle.start_point, [1.0, 0.0, 0.0])
    assert np.allclose(particle.end_point, [0.0, 0.0, 0.0])
    assert np.allclose(particle.start_dir, [-1.0, -0.0, -0.0])


def test_track_extrema_processor_uses_ppn_candidates(monkeypatch):
    monkeypatch.setattr(
        points_mod, "check_track_orientation_ppn", lambda start, end, candidates: True
    )
    particle = _track_particle()
    processor = TrackExtremaProcessor(method="ppn")

    processor.process({"reco_particles": [particle], "ppn_candidates": [object()]})

    assert np.allclose(particle.start_point, [0.0, 0.0, 0.0])


def test_track_extrema_processor_reports_missing_ppn_candidates():
    processor = TrackExtremaProcessor(method="ppn")

    with pytest.raises(KeyError, match="ppn"):
        processor.process({"reco_particles": [_track_particle()]})


def test_track_extrema_processor_rejects_unknown_method():
    processor = TrackExtremaProcessor(method="bad")

    with pytest.raises(ValueError, match="not recognized"):
        processor.process({"reco_particles": [_track_particle()]})
