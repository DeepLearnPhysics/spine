from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from spine.constants import TRACK_SHP
from spine.post.reco import calo as calo_mod
from spine.post.reco.calo import CalibrationProcessor, CalorimetricEnergyProcessor


def test_calorimetric_energy_processor_applies_scaling_and_shower_fudge():
    track = SimpleNamespace(
        is_truth=False,
        shape=TRACK_SHP,
        depositions=np.asarray([1.0, 2.0], dtype=np.float32),
        calo_ke=-1.0,
    )
    shower = SimpleNamespace(
        is_truth=False,
        shape=0,
        depositions=np.asarray([3.0], dtype=np.float32),
        calo_ke=-1.0,
    )
    processor = CalorimetricEnergyProcessor(
        scaling=cast(Any, "2.0"), shower_fudge=cast(Any, "3.0"), run_mode="reco"
    )

    processor.process({"reco_particles": [track, shower]})

    assert track.calo_ke == 6.0
    assert shower.calo_ke == 18.0


class FakeCalibrationManager:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.calls = []

    def __call__(self, points, depositions, sources=None, run_id=None, track=False):
        self.calls.append(
            {
                "points": points,
                "depositions": depositions,
                "sources": sources,
                "run_id": run_id,
                "track": track,
            }
        )
        offset = 10.0 if track else 1.0
        return depositions + offset


def test_calibration_processor_updates_particles_tensors_and_interactions(monkeypatch):
    monkeypatch.setattr(calo_mod, "CalibrationManager", FakeCalibrationManager)
    reco_particle = SimpleNamespace(
        is_truth=False,
        units="cm",
        shape=TRACK_SHP,
        index=np.array([0, 2], dtype=np.int64),
        points=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        depositions=np.array([1.0, 3.0], dtype=np.float32),
        sources=np.array([[0, 0], [0, 0]], dtype=np.int64),
    )
    truth_particle = SimpleNamespace(
        is_truth=True,
        units="cm",
        shape=0,
        index=np.array([1], dtype=np.int64),
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        depositions_q=np.array([20.0], dtype=np.float32),
        sources=np.array([[0, 1]], dtype=np.int64),
    )
    reco_interaction = SimpleNamespace(
        is_truth=False, index=np.array([0, 1, 2, 3], dtype=np.int64)
    )
    truth_interaction = SimpleNamespace(
        is_truth=True, index=np.array([0, 1, 2], dtype=np.int64)
    )
    data = {
        "run_info": SimpleNamespace(run=7),
        "points": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "depositions": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "sources": np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int64),
        "points_label": np.array(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "depositions_q_label": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "sources_label": np.array([[0, 1], [0, 1], [0, 1]], dtype=np.int64),
        "reco_particles": [reco_particle],
        "truth_particles": [truth_particle],
        "reco_interactions": [reco_interaction],
        "truth_interactions": [truth_interaction],
    }
    processor = CalibrationProcessor(do_tracking=True, run_mode="both", scale=2.0)

    processor.process(data)

    calibrator = cast(FakeCalibrationManager, processor.calibrator)
    np.testing.assert_allclose(reco_particle.depositions, [11.0, 13.0])
    np.testing.assert_allclose(data["depositions"], [11.0, 3.0, 13.0, 5.0])
    np.testing.assert_allclose(reco_interaction.depositions, [11.0, 3.0, 13.0, 5.0])
    np.testing.assert_allclose(truth_particle.depositions_q, [21.0])
    np.testing.assert_allclose(data["depositions_q_label"], [11.0, 21.0, 31.0])
    np.testing.assert_allclose(truth_interaction.depositions_q, [11.0, 21.0, 31.0])
    assert calibrator.cfg == {"scale": 2.0}
    assert any(call["track"] for call in calibrator.calls)
    assert all(call["run_id"] == 7 for call in calibrator.calls)


def test_calibration_processor_skips_empty_particles(monkeypatch):
    monkeypatch.setattr(calo_mod, "CalibrationManager", FakeCalibrationManager)
    particle = SimpleNamespace(
        is_truth=False,
        units="cm",
        shape=TRACK_SHP,
        index=np.empty(0, dtype=np.int64),
        points=np.empty((0, 3), dtype=np.float32),
        depositions=np.empty(0, dtype=np.float32),
        sources=np.empty((0, 2), dtype=np.int64),
    )
    data = {
        "points": np.empty((0, 3), dtype=np.float32),
        "depositions": np.empty(0, dtype=np.float32),
        "sources": np.empty((0, 2), dtype=np.int64),
        "reco_particles": [particle],
        "reco_interactions": [],
    }
    processor = CalibrationProcessor(run_mode="reco")

    processor.process(data)

    assert len(cast(FakeCalibrationManager, processor.calibrator).calls) == 1
