import numpy as np
import pytest

import spine.calib.transparency as transparency_module
from spine.calib.transparency import TransparencyCalibrator


def test_transparency_calibrator_applies_map_correction(transparency_db):
    calibrator = TransparencyCalibrator(str(transparency_db), num_tpcs=4)

    corrected = calibrator.process(
        np.array([[0.0, 1.25, 1.25]]),
        np.array([12.0]),
        tpc_id=3,
        run_id=100,
    )

    assert np.allclose(corrected, [2.0])


def test_transparency_calibrator_static_run_overrides_event_run(transparency_db):
    calibrator = TransparencyCalibrator(str(transparency_db), num_tpcs=4, run_id=100)

    corrected = calibrator.process(
        np.array([[0.0, 0.25, 0.25]]),
        np.array([8.0]),
        tpc_id=1,
        run_id=None,
    )

    assert np.allclose(corrected, [4.0])


def test_transparency_calibrator_requires_run_id(transparency_db):
    calibrator = TransparencyCalibrator(str(transparency_db), num_tpcs=4)

    with pytest.raises(ValueError, match="Must provide a run ID"):
        calibrator.process(np.zeros((1, 3)), np.ones(1), tpc_id=0, run_id=None)


def test_transparency_calibrator_loads_static_root_maps(monkeypatch):
    fake_root = FakeROOT()
    monkeypatch.setattr(transparency_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(transparency_module, "ROOT", fake_root)
    calibrator = TransparencyCalibrator(
        num_tpcs=2,
        transparency_file="maps.root",
        map_pattern="correction_{plane_id}_{tpc_id}",
        plane_id=2,
    )

    corrected = calibrator.process(
        np.array([[0.0, -0.5, 10.5], [0.0, 0.5, 10.5]]),
        np.array([3.0, 3.0]),
        tpc_id=0,
        run_id=None,
    )

    assert fake_root.opened == ("maps.root", "r")
    assert fake_root.file.closed
    assert np.allclose(corrected, [6.0, 3.0])


def test_transparency_calibrator_can_divide_root_deviations(monkeypatch):
    fake_root = FakeROOT()
    monkeypatch.setattr(transparency_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(transparency_module, "ROOT", fake_root)
    calibrator = TransparencyCalibrator(
        num_tpcs=2,
        transparency_file="maps.root",
        map_pattern="correction_{plane_id}_{tpc_id}",
        map_type="deviation",
    )

    corrected = calibrator.process(
        np.array([[0.0, -0.5, 10.5]]),
        np.array([3.0]),
        tpc_id=0,
        run_id=None,
    )

    assert np.allclose(corrected, [1.5])


def test_transparency_calibrator_can_multiply_database_corrections(transparency_db):
    calibrator = TransparencyCalibrator(
        str(transparency_db),
        num_tpcs=4,
        map_type="correction",
    )

    corrected = calibrator.process(
        np.array([[0.0, 1.25, 1.25]]),
        np.array([12.0]),
        tpc_id=3,
        run_id=100,
    )

    assert np.allclose(corrected, [72.0])


def test_transparency_calibrator_requires_exactly_one_source(transparency_db):
    with pytest.raises(ValueError, match="number of TPCs"):
        TransparencyCalibrator(transparency_file="maps.root")

    with pytest.raises(ValueError, match="exactly one"):
        TransparencyCalibrator(num_tpcs=2)

    with pytest.raises(ValueError, match="exactly one"):
        TransparencyCalibrator(
            str(transparency_db),
            num_tpcs=2,
            transparency_file="maps.root",
        )

    with pytest.raises(ValueError, match="map type"):
        TransparencyCalibrator(
            str(transparency_db),
            num_tpcs=2,
            map_type="unknown",
        )


def test_transparency_calibrator_requires_root_for_static_maps(monkeypatch):
    monkeypatch.setattr(transparency_module, "ROOT_AVAILABLE", False)
    with pytest.raises(ImportError, match="ROOT"):
        TransparencyCalibrator(num_tpcs=2, transparency_file="maps.root")


def test_transparency_calibrator_rejects_bad_root_file(monkeypatch):
    fake_root = FakeROOT()
    fake_root.file = None
    monkeypatch.setattr(transparency_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(transparency_module, "ROOT", fake_root)

    with pytest.raises(OSError, match="Could not open"):
        TransparencyCalibrator(num_tpcs=2, transparency_file="missing.root")


def test_transparency_calibrator_rejects_missing_root_map(monkeypatch):
    fake_root = FakeROOT()
    del fake_root.file.hists["correction_2_1"]
    monkeypatch.setattr(transparency_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(transparency_module, "ROOT", fake_root)

    with pytest.raises(KeyError, match="correction_2_1"):
        TransparencyCalibrator(
            num_tpcs=2,
            transparency_file="maps.root",
            map_pattern="correction_{plane_id}_{tpc_id}",
        )

    assert fake_root.file.closed


class FakeAxis:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def GetXmin(self):
        return self.low

    def GetXmax(self):
        return self.high


class FakeTH2:
    def __init__(self, values):
        self.values = np.asarray(values)

    def GetXaxis(self):
        return FakeAxis(10.0, 14.0)

    def GetYaxis(self):
        return FakeAxis(-1.0, 1.0)

    def GetNbinsX(self):
        return self.values.shape[0]

    def GetNbinsY(self):
        return self.values.shape[1]

    def GetBinContent(self, ix, iy):
        return self.values[ix - 1, iy - 1]


class FakeTFile:
    def __init__(self):
        self.closed = False
        self.hists = {
            "correction_2_0": FakeTH2([[2.0, 0.0], [4.0, 8.0]]),
            "correction_2_1": FakeTH2([[3.0, 6.0], [9.0, 12.0]]),
        }

    def IsZombie(self):
        return False

    def Get(self, name):
        return self.hists.get(name)

    def Close(self):
        self.closed = True


class FakeTFileFactory:
    def __init__(self, root):
        self.root = root

    def Open(self, path, mode):
        self.root.opened = (path, mode)
        return self.root.file


class FakeROOT:
    def __init__(self):
        self.opened = None
        self.file = FakeTFile()
        self.TFile = FakeTFileFactory(self)
