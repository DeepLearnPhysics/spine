from types import SimpleNamespace

import numpy as np
import pytest

import spine.calib.manager as manager_mod
from spine.calib.field import FieldMap
from spine.data.out import RecoInteraction, RecoParticle
from spine.post.reco.calo import CalibrationProcessor


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
