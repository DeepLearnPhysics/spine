from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from spine.constants import PION_PID, PROT_PID, SHOWR_SHP, TRACK_SHP
from spine.post.reco import shower as shower_mod


class FakeTPC:
    def __init__(self):
        self.modules = [
            SimpleNamespace(boundaries=np.array([[0.0, 1.0]] * 3)),
            SimpleNamespace(boundaries=np.array([[1.0, 2.0]] * 3)),
        ]
        self.chambers = [
            SimpleNamespace(boundaries=np.array([[0.0, 1.0]] * 3)),
            SimpleNamespace(boundaries=np.array([[1.0, 2.0]] * 3)),
        ]
        self.num_modules = 2
        self.num_chambers = 2


class FakeGeo:
    def __init__(self):
        self.boundaries = np.array([[0.0, 10.0]] * 3)
        self.tpc = FakeTPC()

    def get_closest_module(self, points):
        return np.asarray(points[:, 0] > 0.5, dtype=np.int64)

    def get_closest_tpc(self, points):
        return np.asarray(points[:, 0] > 0.5, dtype=np.int64)


class FakeFitter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def fit(self, **kwargs):
        self.calls.append(kwargs)
        return float(np.sum(kwargs["reco_box_energy"]) + 1.0)


def make_particle(**overrides):
    data = {
        "id": 0,
        "shape": SHOWR_SHP,
        "pid": 0,
        "is_primary": True,
        "is_truth": False,
        "units": "cm",
        "points": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "depositions": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "start_point": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "start_dir": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "length": 1.0,
        "calo_ke": 6.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_shower_parametric_energy_processor_validates_and_fits(monkeypatch):
    monkeypatch.setattr(shower_mod.GeoManager, "get_instance", lambda: FakeGeo())
    monkeypatch.setattr(shower_mod, "ShowerEnergyFitter", FakeFitter)

    with pytest.raises(ValueError, match="geometry mode"):
        shower_mod.ShowerParametricEnergyProcessor(mode="volume")

    shower = make_particle()
    low_energy = make_particle(calo_ke=0.5)
    track = make_particle(shape=TRACK_SHP, calo_ke=6.0)
    processor = shower_mod.ShowerParametricEnergyProcessor(
        mode="module", energy_bounds=(1.0, 100.0)
    )

    processor.process({"reco_particles": [shower, low_energy, track]})

    assert shower.calo_ke == pytest.approx(7.0)
    assert low_energy.calo_ke == 0.5
    assert track.calo_ke == 6.0
    fitter = cast(FakeFitter, processor.fitter)
    np.testing.assert_allclose(fitter.calls[0]["reco_box_energy"], [1.0, 5.0])


def test_shower_parametric_energy_processor_detector_mode(monkeypatch):
    monkeypatch.setattr(shower_mod.GeoManager, "get_instance", lambda: FakeGeo())
    monkeypatch.setattr(shower_mod, "ShowerEnergyFitter", FakeFitter)
    shower = make_particle(calo_ke=4.0)
    processor = shower_mod.ShowerParametricEnergyProcessor(mode="detector")

    processor.process({"reco_particles": [shower]})

    assert shower.calo_ke == pytest.approx(5.0)


def test_shower_parametric_energy_processor_tpc_mode(monkeypatch):
    monkeypatch.setattr(shower_mod.GeoManager, "get_instance", lambda: FakeGeo())
    monkeypatch.setattr(shower_mod, "ShowerEnergyFitter", FakeFitter)
    shower = make_particle()
    processor = shower_mod.ShowerParametricEnergyProcessor(mode="tpc")

    processor.process({"reco_particles": [shower]})

    assert shower.calo_ke == pytest.approx(7.0)


def test_shower_conversion_distance_processor_rejects_corrupt_mode():
    shower = make_particle()
    inter = SimpleNamespace(
        vertex=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        particles=[shower],
    )
    processor = shower_mod.ShowerConversionDistanceProcessor(mode="vertex_to_start")
    processor.mode = "corrupt"

    with pytest.raises(ValueError, match="not recognized"):
        processor.process({"reco_interactions": [inter]})


def test_shower_conversion_distance_processor_modes():
    with pytest.raises(AssertionError, match="Conversion distance"):
        shower_mod.ShowerConversionDistanceProcessor(mode="bad")

    shower = make_particle()
    proton = make_particle(
        shape=TRACK_SHP,
        pid=PROT_PID,
        points=np.array([[3.0, 0.0, 0.0]], dtype=np.float32),
    )
    pion = make_particle(
        shape=TRACK_SHP,
        pid=PION_PID,
        points=np.array([[2.5, 0.0, 0.0]], dtype=np.float32),
    )
    secondary = make_particle(is_primary=False)
    inter = SimpleNamespace(
        vertex=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        particles=[shower, proton, pion, secondary],
    )

    processor = shower_mod.ShowerConversionDistanceProcessor(mode="vertex_to_start")
    processor.process({"reco_interactions": [inter]})
    assert shower.vertex_distance == pytest.approx(1.0)

    processor = shower_mod.ShowerConversionDistanceProcessor(mode="vertex_to_points")
    processor.process({"reco_interactions": [inter]})
    assert shower.vertex_distance == pytest.approx(0.0)

    processor = shower_mod.ShowerConversionDistanceProcessor(mode="protons_to_points")
    processor.process({"reco_interactions": [inter]})
    assert shower.vertex_distance == pytest.approx(0.5)

    no_proton_inter = SimpleNamespace(vertex=inter.vertex, particles=[shower])
    assert processor.get_protons_to_points(no_proton_inter, shower) == -1


def test_shower_start_merge_processor_merges_tracks(monkeypatch):
    monkeypatch.setattr(shower_mod, "cluster_dedx", lambda *args, **kwargs: 1.0)
    shower = make_particle(id=0)
    merged_track = make_particle(
        id=1,
        shape=TRACK_SHP,
        points=np.array([[0.2, 0.0, 0.0]], dtype=np.float32),
        start_point=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        length=1.0,
    )
    long_track = make_particle(id=2, shape=TRACK_SHP, length=100.0)
    secondary_track = make_particle(id=3, shape=TRACK_SHP, is_primary=False)
    shower.merged = []
    shower.merge = lambda part: shower.merged.append(part.id)
    inter = SimpleNamespace(
        particles=[shower, merged_track, long_track, secondary_track],
        leading_shower=shower,
    )
    no_shower = SimpleNamespace(
        particles=[make_particle(id=5, shape=TRACK_SHP)],
        leading_shower=None,
    )
    processor = shower_mod.ShowerStartMergeProcessor(track_dedx_limit=2.0)

    result = processor.process({"reco_interactions": [inter, no_shower]})

    assert shower.merged == [1]
    assert [part.id for part in inter.particles] == [0, 1, 2]
    assert np.array_equal(inter.particle_ids, np.array([0, 1, 2]))
    assert len(result["reco_particles"]) == 4


def test_shower_start_merge_processor_rejects_bad_candidates(monkeypatch):
    monkeypatch.setattr(shower_mod, "cluster_dedx", lambda *args, **kwargs: 10.0)
    processor = shower_mod.ShowerStartMergeProcessor(track_dedx_limit=2.0)
    shower = make_particle()
    track = make_particle(shape=TRACK_SHP)

    assert processor.check_merge(shower, track) is False

    processor = shower_mod.ShowerStartMergeProcessor(angle_threshold=0.1)
    track.start_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert processor.check_merge(shower, track) is False

    processor = shower_mod.ShowerStartMergeProcessor(distance_threshold=0.1)
    track.start_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    track.points = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
    assert processor.check_merge(shower, track) is False


def test_shower_start_correction_processor_updates_start_and_direction(monkeypatch):
    monkeypatch.setattr(
        shower_mod,
        "cluster_direction",
        lambda points, start, max_dist, optimize: np.array([0.0, 1.0, 0.0]),
    )
    shower = make_particle(
        points=np.array([[5.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        start_point=np.array([5.0, 0.0, 0.0], dtype=np.float32),
    )
    track = make_particle(shape=TRACK_SHP, points=np.array([[0.0, 0.0, 0.0]]))
    secondary = make_particle(is_primary=False)
    inter = SimpleNamespace(particles=[shower, track, secondary])
    processor = shower_mod.ShowerStartCorrectionProcessor(radius=2, optimize=False)

    processor.process({"reco_interactions": [inter]})

    np.testing.assert_allclose(shower.start_point, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(shower.start_dir, [0.0, 1.0, 0.0])

    no_track = SimpleNamespace(particles=[shower])
    assert np.array_equal(
        processor.correct_shower_start(no_track, shower), shower.start_point
    )
