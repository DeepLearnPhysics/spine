from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import spine.calib.manager as manager_mod
from spine.calib.field import FieldMap
from spine.constants import TRACK_SHP
from spine.data.out import RecoInteraction, RecoParticle
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
        self.update_points = False

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
        return points, depositions + offset


class FakePointCalibrationManager(FakeCalibrationManager):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.update_points = True

    def __call__(self, points, depositions, sources=None, run_id=None, track=False):
        _, calibrated_depositions = super().__call__(
            points, depositions, sources, run_id, track
        )
        return points + np.array([2.0, 0.0, 0.0]), calibrated_depositions


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


def test_calibration_processor_updates_truth_points_and_interactions(monkeypatch):
    monkeypatch.setattr(calo_mod, "CalibrationManager", FakePointCalibrationManager)
    particle = SimpleNamespace(
        is_truth=True,
        units="cm",
        shape=0,
        index=np.array([0], dtype=np.int64),
        points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        depositions_q=np.array([10.0], dtype=np.float32),
        sources=np.array([[0, 0]], dtype=np.int64),
    )
    interaction = SimpleNamespace(
        is_truth=True,
        index=np.array([0, 1], dtype=np.int64),
    )
    data = {
        "points_label": np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
        "depositions_q_label": np.array([10.0, 20.0], dtype=np.float32),
        "sources_label": np.array([[0, 0], [0, 0]], dtype=np.int64),
        "truth_particles": [particle],
        "truth_interactions": [interaction],
    }
    processor = CalibrationProcessor(run_mode="truth")

    processor.process(data)

    expected_points = np.array([[3.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(particle.points, expected_points[[0]])
    np.testing.assert_allclose(interaction.points, expected_points)
    np.testing.assert_allclose(data["points_label"], expected_points)


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


class FakeTPC:
    def __init__(self, anode_pos, center):
        self.anode_pos = anode_pos
        self.drift_axis = 0
        self.drift_dir = np.array([1.0, 0.0, 0.0])
        self.center = np.asarray(center, dtype=float)
        self.dimensions = np.asarray((10.0, 10.0, 10.0), dtype=float)
        self.boundaries = np.vstack(
            (self.center - self.dimensions / 2.0, self.center + self.dimensions / 2.0)
        ).T


class FakeTPCSet:
    num_modules = 1
    num_chambers_per_module = 2
    num_chambers = 2

    def __init__(self):
        self._tpcs = [
            [
                FakeTPC(0.0, center=(0.0, 0.0, 0.0)),
                FakeTPC(10.0, center=(10.0, 0.0, 0.0)),
            ]
        ]
        self.chambers = [tpc for module in self._tpcs for tpc in module]
        lower = np.min(
            np.vstack([tpc.boundaries[:, 0] for tpc in self.chambers]), axis=0
        )
        upper = np.max(
            np.vstack([tpc.boundaries[:, 1] for tpc in self.chambers]), axis=0
        )
        self.modules = [
            SimpleNamespace(
                center=(lower + upper) / 2.0,
                dimensions=upper - lower,
                boundaries=np.vstack((lower, upper)).T,
            )
        ]

    def __getitem__(self, index):
        return self._tpcs[index]


class FakeGeo:
    def __init__(self):
        self.tpc = FakeTPCSet()

    def get_volume_index(self, sources, module_id, tpc_id):
        return np.where((sources[:, 0] == module_id) & (sources[:, 1] == tpc_id))[0]

    def get_closest_tpc_indexes(self, points):
        return [np.where(points[:, 0] <= 5.0)[0], np.where(points[:, 0] > 5.0)[0]]

    def translate(self, points, source_module, target_module):
        return points + float(target_module - source_module)


@pytest.fixture
def fake_geo():
    return FakeGeo()


def test_calibration_processor_updates_field_corrected_points(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [2.0, 0.0, 0.0], dtype=float),
        [[0.0, 10.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    processor = CalibrationProcessor(
        obj_type=("particle", "interaction"),
        field={"field_map": field_map},
    )

    points = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    depositions = np.array([10.0, 20.0])
    sources = np.array([[0, 0], [0, 0]])
    particle = RecoParticle(
        index=np.array([0], dtype=np.int32),
        points=points[[0]].copy(),
        depositions=depositions[[0]].copy(),
        sources=sources[[0]].copy(),
    )
    interaction = RecoInteraction(
        index=np.array([0, 1], dtype=np.int32),
        points=points.copy(),
        depositions=depositions.copy(),
        sources=sources.copy(),
    )
    data = {
        "points": points.copy(),
        "depositions": depositions.copy(),
        "sources": sources,
        "reco_particles": [particle],
        "reco_interactions": [interaction],
    }

    processor.process(data)

    expected_points = points + np.array([2.0, 0.0, 0.0])
    assert np.allclose(data["points"], expected_points)
    assert np.allclose(particle.points, expected_points[[0]])
    assert np.allclose(interaction.points, expected_points)
