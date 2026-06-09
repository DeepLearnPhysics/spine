import numpy as np
import pytest

import spine.calib.manager as manager_mod
from spine.calib.gain import GainCalibrator
from spine.calib.manager import CalibrationManager
from spine.calib.recombination import RecombinationCalibrator


class FakeMeta:
    def to_cm(self, points, center=True):
        return points + 1.0


def test_manager_parses_labels_and_explicit_names(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)

    manager = CalibrationManager(
        first_gain={"name": "gain", "gain": 2.0},
        recomb={"name": "recombination", "efield": 0.5},
    )

    assert list(manager.modules) == ["first_gain", "recomb"]
    assert isinstance(manager.modules["first_gain"], GainCalibrator)
    assert isinstance(manager.modules["recomb"], RecombinationCalibrator)
    assert manager.modules["recomb"].drift_dir.tolist() == [1.0, 0.0, 0.0]


def test_manager_requires_gain_when_recombination_is_configured(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)

    with pytest.raises(ValueError, match="Must provide gain"):
        CalibrationManager(recombination={"efield": 0.5})

    with pytest.raises(ValueError, match="Must provide gain"):
        CalibrationManager(recomb={"name": "recombination", "efield": 0.5})

    with pytest.raises(ValueError, match="before recombination"):
        CalibrationManager(
            recomb={"name": "recombination", "efield": 0.5},
            gain={"gain": 2.0},
        )


def test_manager_applies_modules_in_config_order(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(first_gain={"name": "gain", "gain": [2.0, 3.0]})
    points = np.array([[1.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    values = np.array([10.0, 10.0])
    sources = np.array([[0, 0], [0, 1]])

    corrected = manager(points, values, sources=sources)

    assert np.allclose(corrected, [20.0, 30.0])


def test_manager_can_infer_tpc_indexes_without_sources(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(gain={"gain": [2.0, 3.0]})

    corrected = manager(
        np.array([[1.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        np.array([10.0, 10.0]),
    )

    assert np.allclose(corrected, [20.0, 30.0])


def test_manager_applies_meta_module_translation_and_empty_tpc(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(gain={"gain": [2.0, 3.0]})

    corrected = manager(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([10.0]),
        sources=np.array([[0, 0]]),
        meta=FakeMeta(),
        module_id=1,
    )

    assert np.allclose(corrected, [20.0])


def test_manager_dispatches_lifetime_and_unknown_modules(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(lifetime={"lifetime": 10.0, "driftv": 2.0})

    corrected = manager(
        np.array([[4.0, 0.0, 0.0]]),
        np.array([1.0]),
        sources=np.array([[0, 0]]),
    )
    assert np.allclose(corrected, [np.exp(4.0 / 20.0)])

    manager.module_names["lifetime"] = "unknown"
    with pytest.raises(ValueError, match="not recognized"):
        manager(
            np.array([[4.0, 0.0, 0.0]]), np.array([1.0]), sources=np.array([[0, 0]])
        )


def test_manager_validates_points_without_sources(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(gain={"gain": 2.0})

    with pytest.raises(ValueError, match="must provide points"):
        manager(None, np.array([1.0]))  # type: ignore[arg-type]


def test_manager_dispatches_recombination(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(
        gain_applied=True,
        recombination={"efield": 0.5},
    )

    corrected = manager(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([1000.0]),
        sources=np.array([[0, 0]]),
        track=False,
    )

    assert corrected[0] > 0.0


def test_manager_dispatches_transparency(monkeypatch, fake_geo, transparency_db):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(
        transparency={"transparency_db": str(transparency_db), "run_id": 100}
    )

    corrected = manager(
        np.array([[0.0, 1.25, 1.25]]),
        np.array([12.0]),
        sources=np.array([[0, 1]]),
    )

    assert np.allclose(corrected, [3.0])
