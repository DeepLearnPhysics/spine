from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import spine.post.reco.tracking as tracking_mod
from spine.constants import MUON_PID, PION_PID, TRACK_SHP
from spine.post.reco.tracking import CSDAEnergyProcessor


def _track_particle(points):
    return SimpleNamespace(
        is_truth=False,
        shape=TRACK_SHP,
        pid=MUON_PID,
        units="cm",
        points=np.asarray(points, dtype=np.float32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        length=-1.0,
        csda_ke=-1.0,
        csda_ke_per_pid=np.full(6, -1.0, dtype=np.float32),
    )


def test_csda_energy_processor_assigns_length_and_energy(monkeypatch):
    monkeypatch.setattr(
        tracking_mod,
        "csda_table_spline",
        lambda pid: lambda length: np.asarray(pid + length),
    )
    monkeypatch.setattr(tracking_mod, "get_track_length", lambda *args, **kwargs: 4.0)
    processor = CSDAEnergyProcessor(
        run_mode="reco", include_pids=(MUON_PID, PION_PID), fill_per_pid=True
    )
    particle = _track_particle([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    processor.process({"reco_particles": [particle]})

    assert particle.length == 4.0
    assert particle.csda_ke == MUON_PID + 4.0
    assert particle.csda_ke_per_pid[PION_PID] == PION_PID + 4.0


def test_csda_energy_processor_sets_zero_for_zero_length(monkeypatch):
    monkeypatch.setattr(
        tracking_mod, "csda_table_spline", lambda pid: lambda length: np.asarray(1.0)
    )
    monkeypatch.setattr(tracking_mod, "get_track_length", lambda *args, **kwargs: 0.0)
    processor = CSDAEnergyProcessor(run_mode="reco", fill_per_pid=True)
    particle = _track_particle([[0.0, 0.0, 0.0]])

    processor.process({"reco_particles": [particle]})

    assert particle.csda_ke == 0.0
    assert np.all(particle.csda_ke_per_pid[list(processor.include_pids)] == 0.0)


def test_csda_energy_processor_skips_nontracks_and_empty_tracks(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tracking_mod, "csda_table_spline", lambda pid: lambda length: np.asarray(1.0)
    )
    monkeypatch.setattr(
        tracking_mod, "get_track_length", lambda *args, **kwargs: calls.append(args)
    )
    processor = CSDAEnergyProcessor(run_mode="reco")
    nontrack = _track_particle([[0.0, 0.0, 0.0]])
    nontrack.shape = 0
    empty = _track_particle([])

    processor.process({"reco_particles": [nontrack, empty]})

    assert calls == []


def test_csda_energy_processor_assigns_truth_reco_length(monkeypatch):
    monkeypatch.setattr(
        tracking_mod, "csda_table_spline", lambda pid: lambda length: np.asarray(2.0)
    )
    monkeypatch.setattr(tracking_mod, "get_track_length", lambda *args, **kwargs: 3.0)
    processor = CSDAEnergyProcessor(run_mode="truth")
    particle = _track_particle([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    particle.is_truth = True
    particle.reco_length = -1.0
    particle.reco_ke = -1.0

    processor.process({"truth_particles": [particle]})

    assert particle.reco_length == 3.0
    assert particle.csda_ke == 2.0
