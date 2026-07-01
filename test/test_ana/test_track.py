from __future__ import annotations

import numpy as np
import pytest

from spine.ana.diag.track import TrackCompletenessAna
from spine.constants import MUON_PID, TRACK_SHP


class FakeMeta:
    size = np.array([1.0, 1.0, 1.0])


class FakeParticle:
    id = 7
    shape = TRACK_SHP
    pid = MUON_PID
    start_point = np.array([0.0, 0.0, 0.0])
    end_point = np.array([5.0, 0.0, 0.0])
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )
    is_truth = False


@pytest.fixture(autouse=True)
def _disable_writers(monkeypatch):
    monkeypatch.setattr(
        TrackCompletenessAna, "initialize_writer", lambda self, name: None
    )


def test_track_completeness_ana_validates_configuration():
    with pytest.raises(ValueError, match="two values"):
        TrackCompletenessAna(time_window=(0.0, 1.0, 2.0), run_mode="truth")

    with pytest.raises(ValueError, match="reconstructed particle"):
        TrackCompletenessAna(time_window=(0.0, 1.0), run_mode="both")


def test_track_completeness_ana_process_writes_gap_summary(monkeypatch):
    rows = []
    monkeypatch.setattr(
        TrackCompletenessAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    ana = TrackCompletenessAna(run_mode="reco", length_threshold=None)

    ana.process({"meta": FakeMeta(), "reco_particles": [FakeParticle()]})

    assert len(rows) == 1
    name, row = rows[0]
    assert name == "reco"
    assert row["particle_id"] == 7
    assert row["num_gaps"] == 1
    assert row["gap_length"] > 0.0
    assert row["gap_frac"] > 0.0


def test_track_completeness_ana_skips_filtered_particles(monkeypatch):
    rows = []
    monkeypatch.setattr(
        TrackCompletenessAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )
    wrong_shape = FakeParticle()
    wrong_shape.shape = 0
    wrong_pid = FakeParticle()
    wrong_pid.pid = 999
    too_short = FakeParticle()
    ana = TrackCompletenessAna(run_mode="reco", length_threshold=100.0)

    ana.process(
        {
            "meta": FakeMeta(),
            "reco_particles": [wrong_shape, wrong_pid, too_short],
        }
    )

    assert rows == []


def test_track_completeness_ana_skips_out_of_time_truth_particle(monkeypatch):
    rows = []
    monkeypatch.setattr(
        TrackCompletenessAna,
        "append",
        lambda self, name, **kwargs: rows.append((name, kwargs)),
    )

    class TruthParticle(FakeParticle):
        is_truth = True
        t = 20.0

    ana = TrackCompletenessAna(run_mode="truth", time_window=(0.0, 10.0))

    ana.process({"meta": FakeMeta(), "truth_particles": [TruthParticle()]})

    assert rows == []


def test_track_completeness_static_helpers():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )
    labels = TrackCompletenessAna.cluster_track_chunks(
        points, points[0], points[-1], pixel_size=1.0
    )
    gaps = TrackCompletenessAna.sequential_cluster_distances(points, labels, points[0])

    assert labels.tolist() == [0, 0, 1, 1]
    assert np.allclose(gaps, [3.0])


def test_track_completeness_single_cluster_has_no_gaps():
    labels = np.zeros(2, dtype=np.int32)
    gaps = TrackCompletenessAna.sequential_cluster_distances(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        labels,
        np.zeros(3),
    )

    assert len(gaps) == 0


def test_track_completeness_rejects_non_cubic_pixels(monkeypatch):
    class BadMeta:
        size = np.array([1.0, 2.0, 1.0])

    ana = TrackCompletenessAna(run_mode="reco")

    with pytest.raises(ValueError, match="Non-cubic"):
        ana.process({"meta": BadMeta(), "reco_particles": []})
