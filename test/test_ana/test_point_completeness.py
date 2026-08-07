from __future__ import annotations

import numpy as np
import pytest

from spine.ana.diag.point import PointCompletenessAna
from spine.ana.factories import ANA_DICT
from spine.data.out import TruthParticle


class FakeMeta:
    size = np.array([1.0, 1.0, 1.0])


@pytest.fixture(autouse=True)
def _disable_writers(monkeypatch):
    monkeypatch.setattr(
        PointCompletenessAna,
        "initialize_writer",
        lambda self, name: None,
    )


def test_point_completeness_validates_configuration():
    assert ANA_DICT["point_completeness"] is PointCompletenessAna
    assert ANA_DICT["point_metrics"] is PointCompletenessAna

    with pytest.raises(ValueError, match="two values"):
        PointCompletenessAna(time_window=(0.0, 1.0, 2.0))

    with pytest.raises(ValueError, match="lower bound"):
        PointCompletenessAna(time_window=(1.0, 0.0))

    with pytest.raises(ValueError, match="finite and positive"):
        PointCompletenessAna(match_distance=0.0)


def test_point_completeness_processes_truth_particles(monkeypatch):
    rows = []
    monkeypatch.setattr(
        PointCompletenessAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    particle = TruthParticle(
        id=3,
        shape=1,
        t=5.0,
        points=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        points_g4=np.array([[0.5, 0.0, 0.0], [20.0, 0.0, 0.0]]),
    )
    ana = PointCompletenessAna(
        time_window=(0.0, 10.0),
        match_distance=1.0,
    )

    assert ana.keys["points_g4"] is True
    assert ana.keys["meta"] is True
    ana.process({"meta": FakeMeta(), "truth_particles": [particle]})

    assert rows == [
        (
            "particle",
            {
                "id": 3,
                "shape": 1,
                "num_points": 2,
                "num_points_g4": 2,
                "purity": 0.5,
                "efficiency": 0.5,
            },
        )
    ]


def test_point_completeness_uses_voxel_diagonal(monkeypatch):
    rows = []
    monkeypatch.setattr(
        PointCompletenessAna,
        "append",
        lambda self, name, **kwargs: rows.append(kwargs),
    )
    particle = TruthParticle(
        points=np.array([[0.0, 0.0, 0.0]]),
        points_g4=np.array([[1.5, 0.0, 0.0]]),
    )
    ana = PointCompletenessAna()

    ana.process({"meta": FakeMeta(), "truth_particles": [particle]})

    assert rows[0]["purity"] == 1.0
    assert rows[0]["efficiency"] == 1.0


def test_point_completeness_rejects_invalid_voxel_diagonal():
    """Metadata-derived matching requires finite, positive voxel sizes."""

    class InvalidMeta:
        size = np.zeros(3)

    ana = PointCompletenessAna()

    with pytest.raises(ValueError, match="positive voxel sizes"):
        ana.process({"meta": InvalidMeta(), "truth_particles": []})


def test_point_completeness_records_empty_point_sets(monkeypatch):
    rows = []
    monkeypatch.setattr(
        PointCompletenessAna,
        "append",
        lambda self, name, **kwargs: rows.append(kwargs),
    )
    particles = [
        TruthParticle(
            points=np.empty((0, 3)),
            points_g4=np.array([[0.0, 0.0, 0.0]]),
        ),
        TruthParticle(
            points=np.array([[0.0, 0.0, 0.0]]),
            points_g4=np.empty((0, 3)),
        ),
    ]
    ana = PointCompletenessAna(match_distance=1.0)

    ana.process({"meta": FakeMeta(), "truth_particles": particles})

    assert len(rows) == 2
    assert np.isnan(rows[0]["purity"])
    assert rows[0]["efficiency"] == 0.0
    assert rows[1]["purity"] == 0.0
    assert np.isnan(rows[1]["efficiency"])


def test_point_completeness_filters_out_of_time_objects(monkeypatch):
    rows = []
    monkeypatch.setattr(
        PointCompletenessAna,
        "append",
        lambda self, name, **kwargs: rows.append(kwargs),
    )
    particle = TruthParticle(
        t=20.0,
        points=np.array([[0.0, 0.0, 0.0]]),
        points_g4=np.array([[0.0, 0.0, 0.0]]),
    )
    ana = PointCompletenessAna(
        time_window=(0.0, 10.0),
        match_distance=1.0,
    )

    ana.process({"meta": FakeMeta(), "truth_particles": [particle]})

    assert rows == []


def test_point_completeness_match_fraction():
    source = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    target = np.array([[0.5, 0.0, 0.0]])

    assert PointCompletenessAna.match_fraction(source, target, 1.0) == 0.5
    assert PointCompletenessAna.match_fraction(source, np.empty((0, 3)), 1.0) == 0.0
    assert np.isnan(PointCompletenessAna.match_fraction(np.empty((0, 3)), target, 1.0))
