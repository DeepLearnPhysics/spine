from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import spine.post.reco.pid as pid_mod
from spine.constants import TRACK_SHP
from spine.post.reco.pid import PIDTemplateProcessor


class FakeIdentifier:
    include_pids = (1, 3)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, points, values, end_point, start_point):
        return 3, np.asarray([1.5, 0.25], dtype=np.float32)


def _track_particle(points):
    return SimpleNamespace(
        is_truth=False,
        shape=TRACK_SHP,
        units="cm",
        points=np.asarray(points, dtype=np.float32),
        depositions=np.ones(len(points), dtype=np.float32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        end_point=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        chi2_pid=-1,
        chi2_per_pid=np.full(5, -1.0, dtype=np.float32),
    )


def test_pid_template_processor_assigns_pid_and_scores(monkeypatch):
    monkeypatch.setattr(pid_mod, "TemplateParticleIdentifier", FakeIdentifier)
    processor = PIDTemplateProcessor(run_mode="reco", fill_per_pid=True)
    particle = _track_particle([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    processor.process({"reco_particles": [particle]})

    assert particle.chi2_pid == 3
    assert particle.chi2_per_pid[1] == 1.5
    assert particle.chi2_per_pid[3] == 0.25


def test_pid_template_processor_skips_empty_tracks(monkeypatch):
    monkeypatch.setattr(pid_mod, "TemplateParticleIdentifier", FakeIdentifier)
    processor = PIDTemplateProcessor(run_mode="reco")
    particle = _track_particle([])

    processor.process({"reco_particles": [particle]})

    assert particle.chi2_pid == -1


def test_pid_template_processor_skips_non_tracks(monkeypatch):
    monkeypatch.setattr(pid_mod, "TemplateParticleIdentifier", FakeIdentifier)
    processor = PIDTemplateProcessor(run_mode="reco")
    particle = _track_particle([[0.0, 0.0, 0.0]])
    particle.shape = 0

    processor.process({"reco_particles": [particle]})

    assert particle.chi2_pid == -1
