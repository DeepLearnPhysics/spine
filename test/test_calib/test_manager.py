from types import SimpleNamespace

import numpy as np
import pytest

import spine.calib.field as field_mod
import spine.calib.manager as manager_mod
from spine.calib.field import FieldMap
from spine.calib.gain import GainCalibrator
from spine.calib.manager import CalibrationManager
from spine.calib.recombination import RecombinationCalibrator
from spine.calib.response import ResponseCalibrator


class FakeMeta:
    def to_cm(self, points, center=True):
        return points + 1.0

    def to_px(self, points, floor=False):
        points = points - 1.0
        return np.floor(points) if floor else points


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

    _, corrected = manager(points, values, sources=sources)

    assert np.allclose(corrected, [20.0, 30.0])


def test_manager_applies_response_between_gain_and_recombination(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(
        gain={"gain": 2.0},
        response={"response_func": "x + 1"},
        recombination={"efield": 0.5},
    )

    assert list(manager.modules) == ["gain", "response", "recombination"]
    assert isinstance(manager.modules["response"], ResponseCalibrator)


def test_manager_applies_response_without_other_calibrators(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(response={"response_func": "x**2 + 1"})

    _, corrected = manager(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([3.0]),
        sources=np.array([[0, 0]]),
    )

    assert np.allclose(corrected, [10.0])


def test_manager_can_infer_tpc_indexes_without_sources(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(gain={"gain": [2.0, 3.0]})

    _, corrected = manager(
        np.array([[1.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        np.array([10.0, 10.0]),
    )

    assert np.allclose(corrected, [20.0, 30.0])


def test_manager_applies_meta_module_translation_and_empty_tpc(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(gain={"gain": [2.0, 3.0]})

    _, corrected = manager(
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

    _, corrected = manager(
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

    _, corrected = manager(
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

    _, corrected = manager(
        np.array([[0.0, 1.25, 1.25]]),
        np.array([12.0]),
        sources=np.array([[0, 1]]),
    )

    assert np.allclose(corrected, [3.0])


def test_manager_dispatches_field_before_lifetime(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [2.0, 0.0, 0.0], dtype=float),
        [[0.0, 10.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    manager = CalibrationManager(
        field={"field_map": field_map},
        lifetime={"lifetime": 10.0, "driftv": 2.0},
    )

    _, corrected = manager(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([1.0]),
        sources=np.array([[0, 0]]),
    )

    assert np.allclose(corrected, [np.exp(3.0 / 20.0)])


def test_manager_can_return_field_corrected_points(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [2.0, 0.0, 0.0], dtype=float),
        [[0.0, 10.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    manager = CalibrationManager(field={"field_map": field_map})

    points, values = manager(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([1.0]),
        sources=np.array([[0, 0]]),
    )

    assert np.allclose(points, [[3.0, 0.0, 0.0]])
    assert np.allclose(values, [1.0])


def test_manager_can_field_correct_arbitrary_points(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [2.0, 0.0, 0.0], dtype=float),
        [[0.0, 10.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    manager = CalibrationManager(
        field={"field_map": field_map},
        gain={"gain": [2.0, 3.0]},
    )
    points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    corrected = manager.process_points(points)

    assert corrected.dtype == points.dtype
    np.testing.assert_allclose(corrected, [[3.0, 0.0, 0.0]])


def test_manager_skips_arbitrary_point_correction_without_field(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    manager = CalibrationManager(gain={"gain": [2.0, 3.0]})
    points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    corrected = manager.process_points(points)

    assert corrected is points


@pytest.mark.parametrize("module_id", [None, 0, 1])
def test_manager_restores_input_module_after_field_correction(monkeypatch, module_id):
    """Temporary module translations must be exactly undone before returning."""

    class MultiModuleGeo:
        def __init__(self):
            modules = [
                SimpleNamespace(
                    center=np.array([0.0, 0.0, 0.0]),
                    dimensions=np.array([2.0, 2.0, 2.0]),
                    boundaries=np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
                ),
                SimpleNamespace(
                    center=np.array([100.0, 0.0, 0.0]),
                    dimensions=np.array([2.0, 2.0, 2.0]),
                    boundaries=np.array([[99.0, 101.0], [-1.0, 1.0], [-1.0, 1.0]]),
                ),
            ]
            self.tpc = SimpleNamespace(
                center=np.array([50.0, 0.0, 0.0]),
                modules=modules,
                num_modules=2,
                num_chambers=2,
                num_chambers_per_module=1,
            )

        @staticmethod
        def get_volume_index(sources, module_id, tpc_id):
            return np.where((sources[:, 0] == module_id) & (sources[:, 1] == tpc_id))[0]

        @staticmethod
        def translate(points, source_id, target_id):
            return points + np.array([100.0 * (target_id - source_id), 0.0, 0.0])

    geo = MultiModuleGeo()
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: geo)
    monkeypatch.setattr(field_mod.GeoManager, "get_instance", lambda: geo)
    field_map = FieldMap(
        np.zeros((1, 1, 1, 3), dtype=float),
        [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    manager = CalibrationManager(field={"field_map": field_map})

    points = np.array([[0.0, 0.0, 0.0]])
    source_module = 0 if module_id is None else module_id
    corrected, _ = manager(
        points,
        np.array([1.0]),
        sources=np.array([[source_module, 0]]),
        module_id=module_id,
    )

    np.testing.assert_array_equal(corrected, points)


def test_manager_returns_field_corrected_points_in_input_units(monkeypatch, fake_geo):
    monkeypatch.setattr(manager_mod.GeoManager, "get_instance", lambda: fake_geo)
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [2.0, 0.0, 0.0], dtype=float),
        [[0.0, 10.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    manager = CalibrationManager(field={"field_map": field_map})

    points, values = manager(
        np.array([[1.0, -0.5, -0.5]]),
        np.array([1.0]),
        sources=np.array([[0, 0]]),
        meta=FakeMeta(),
    )

    assert np.allclose(points, [[3.0, -1.0, -1.0]])
    assert np.allclose(values, [1.0])
