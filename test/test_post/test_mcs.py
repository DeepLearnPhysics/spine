import numpy as np
import pytest

import spine.post.reco.mcs as mcs_mod
from spine.constants import MUON_PID, TRACK_SHP
from spine.data.out import RecoParticle
from spine.post.reco.mcs import MCSEnergyProcessor


def test_mcs_validates_configuration():
    with pytest.raises(ValueError, match="tracking algorithm"):
        MCSEnergyProcessor(tracking_mode="bad")

    with pytest.raises(ValueError, match="Angular reconstruction"):
        MCSEnergyProcessor(angle_method="bad")

    with pytest.raises(ValueError, match="fit bounds"):
        MCSEnergyProcessor(lower_bound=100.0, upper_bound=10.0)


def test_mcs_skips_one_point_track():
    processor = MCSEnergyProcessor(run_mode="reco")
    particle = RecoParticle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        start_point=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    processor.process({"reco_particles": [particle]})

    assert np.isnan(particle.mcs_ke)


def test_mcs_assigns_energy_and_pid_hypotheses(monkeypatch):
    fit_kwargs = None

    monkeypatch.setattr(
        mcs_mod,
        "get_track_segments",
        lambda points, segment_length, start_point, **kwargs: (
            [np.asarray([0]), np.asarray([1])],
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.asarray([1.0, 1.0], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(mcs_mod, "mcs_angles", lambda dirs, method: np.asarray([0.1]))

    def fit(*args, **kwargs):
        nonlocal fit_kwargs
        fit_kwargs = kwargs
        return 42.0

    monkeypatch.setattr(mcs_mod, "mcs_fit", fit)
    processor = MCSEnergyProcessor(
        run_mode="reco",
        include_pids=(MUON_PID,),
        fill_per_pid=True,
        lower_bound=20.0,
        upper_bound=5000.0,
        return_invalid=True,
    )
    particle = RecoParticle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )

    processor.process({"reco_particles": [particle]})

    assert particle.mcs_ke == 42.0
    assert particle.mcs_ke_per_pid[MUON_PID] > 0.0
    assert fit_kwargs is not None
    assert fit_kwargs["lower_bound"] == 20.0
    assert fit_kwargs["upper_bound"] == 5000.0
    assert fit_kwargs["return_invalid"] is True


def test_mcs_skips_unwanted_contained_and_angleless_tracks(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mcs_mod,
        "get_track_segments",
        lambda points, segment_length, start_point, **kwargs: (
            None,
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            None,
        ),
    )
    monkeypatch.setattr(mcs_mod, "mcs_angles", lambda dirs, method: np.empty(0))
    monkeypatch.setattr(mcs_mod, "mcs_fit", lambda *args, **kwargs: calls.append(args))
    processor = MCSEnergyProcessor(
        run_mode="reco", include_pids=(MUON_PID,), only_uncontained=True
    )
    nontrack = RecoParticle(
        shape=0,
        pid=MUON_PID,
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    contained = RecoParticle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        is_contained=True,
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    angleless = RecoParticle(
        shape=TRACK_SHP,
        pid=MUON_PID,
        is_contained=False,
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        start_point=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )

    processor.process({"reco_particles": [nontrack, contained, angleless]})

    assert calls == []
