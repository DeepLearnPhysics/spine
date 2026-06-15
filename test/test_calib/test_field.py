from types import SimpleNamespace

import numpy as np
import pytest

from spine.calib import field as field_module
from spine.calib.field import FieldCalibrator, FieldMap


@pytest.fixture
def field_geo(monkeypatch):
    geo = make_two_tpc_geo()
    monkeypatch.setattr(field_module.GeoManager, "get_instance", lambda: geo)
    return geo


def test_field_map_queries_voxels_and_zeros_out_of_bounds_by_default():
    values = np.zeros((2, 2, 2, 3), dtype=float)
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                values[ix, iy, iz] = (ix, iy, iz)

    field_map = FieldMap(values, [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    points = np.asarray(
        [
            [0.25, 0.25, 0.25],
            [1.25, 0.25, 1.75],
            [-1.0, 3.0, 0.25],
        ]
    )

    assert np.allclose(
        field_map.query(points),
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
    )
    assert np.array_equal(field_map.contains(points), [True, True, False])


def test_field_map_validates_inputs():
    values = np.ones((1, 1, 1, 3), dtype=float)
    ranges = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

    with pytest.raises(ValueError, match="Out-of-bounds mode"):
        FieldMap(values, ranges, bounds="bad")

    with pytest.raises(ValueError, match="dense"):
        FieldMap(np.ones((1, 1, 1), dtype=float), ranges)

    with pytest.raises(ValueError, match=r"\[min, max\]"):
        FieldMap(values, [[0.0, 1.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="positive width"):
        FieldMap(values, [[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])


def test_field_map_can_zero_or_raise_out_of_bounds():
    values = np.ones((1, 1, 1, 3), dtype=float)
    points = np.asarray([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5]])

    field_map = FieldMap(values, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], bounds="zero")
    assert np.allclose(field_map.query(points), [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    field_map = FieldMap(values, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], bounds="raise")
    with pytest.raises(IndexError, match="outside"):
        field_map.query(points)


def test_field_map_can_clip_out_of_bounds():
    values = np.zeros((2, 2, 2, 3), dtype=float)
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                values[ix, iy, iz] = (ix, iy, iz)

    field_map = FieldMap(values, [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], bounds="clip")
    assert np.allclose(field_map.query(np.asarray([[-1.0, 3.0, 0.25]])), [[0, 1, 0]])


def test_field_map_edges_and_point_shape_validation():
    field_map = FieldMap(
        np.ones((2, 1, 4, 3), dtype=float),
        [[0.0, 2.0], [-1.0, 1.0], [10.0, 14.0]],
    )

    edges = field_map.edges
    assert np.allclose(edges[0], [0.0, 1.0, 2.0])
    assert np.allclose(edges[1], [-1.0, 1.0])
    assert np.allclose(edges[2], [10.0, 11.0, 12.0, 13.0, 14.0])

    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        field_map.query(np.asarray([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        field_map.contains(np.asarray([0.0, 1.0, 2.0]))


def test_field_calibrator_applies_scaled_displacements(field_geo):
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [1.0, -2.0, 0.5], dtype=float),
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )
    calibrator = FieldCalibrator(field_map=field_map, scale=-1.0)

    points = np.asarray([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    assert np.allclose(
        calibrator.process(points, tpc_id=1),
        [[-0.75, 2.25, -0.25], [-0.25, 2.75, 0.25]],
    )


def test_field_calibrator_leaves_out_of_map_points_unchanged_by_default(field_geo):
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [1.0, -2.0, 0.5], dtype=float),
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )
    calibrator = FieldCalibrator(field_map=field_map)

    points = np.asarray([[0.25, 0.25, 0.25], [2.0, 0.25, 0.25]])
    assert np.allclose(
        calibrator.process(points, tpc_id=1),
        [[1.25, -1.75, 0.75], [2.0, 0.25, 0.25]],
    )


def test_field_calibrator_builds_one_map_per_module(field_geo):
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [1.0, 2.0, 3.0], dtype=float),
        [[0.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    calibrator = FieldCalibrator(field_map=field_map)

    assert len(calibrator.module_maps) == field_geo.tpc.num_modules
    assert calibrator.module_maps[1] is field_map
    assert np.allclose(
        calibrator.module_maps[0].range, [[-2.0, 0.0], [-1.0, 1.0], [-1.0, 1.0]]
    )


def test_field_calibrator_uses_tpc_id_to_pick_module_map(field_geo):
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [1.0, 2.0, 3.0], dtype=float),
        [[0.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    calibrator = FieldCalibrator(field_map=field_map)

    positive_points = np.asarray([[1.0, 0.0, 0.0]])
    negative_points = np.asarray([[-1.0, 0.0, 0.0]])
    assert np.allclose(
        calibrator.process(positive_points, tpc_id=1),
        [[2.0, 2.0, 3.0]],
    )
    assert np.allclose(
        calibrator.process(negative_points, tpc_id=0),
        [[-2.0, 2.0, 3.0]],
    )


def test_field_calibrator_leaves_points_outside_module_map_unchanged(field_geo):
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [1.0, 2.0, 3.0], dtype=float),
        [[0.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    calibrator = FieldCalibrator(field_map=field_map)

    points = np.asarray([[-3.0, 0.0, 0.0]])
    assert np.allclose(calibrator.process(points, tpc_id=0), points)


def test_field_calibrator_requires_tpc_id(field_geo):
    field_map = FieldMap(
        np.full((1, 1, 1, 3), [1.0, 2.0, 3.0], dtype=float),
        [[0.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    calibrator = FieldCalibrator(field_map=field_map)

    with pytest.raises(TypeError, match="tpc_id"):
        calibrator.process(np.asarray([[1.0, 0.0, 0.0]]))


def test_field_calibrator_rejects_map_that_overlaps_no_module(field_geo):
    field_map = FieldMap(
        np.ones((1, 1, 1, 3), dtype=float),
        [[10.0, 12.0], [-1.0, 1.0], [-1.0, 1.0]],
    )

    with pytest.raises(ValueError, match="does not overlap any detector module"):
        FieldCalibrator(field_map=field_map)


def test_field_calibrator_rejects_unmatched_uncovered_module(monkeypatch):
    field_map = FieldMap(
        np.ones((1, 1, 1, 3), dtype=float),
        [[0.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]],
    )
    geo = make_two_tpc_geo()
    geo.tpc.modules[0].dimensions = np.asarray([3.0, 2.0, 2.0])
    monkeypatch.setattr(field_module.GeoManager, "get_instance", lambda: geo)

    with pytest.raises(ValueError, match="No equivalent covered module"):
        FieldCalibrator(field_map=field_map)


def test_field_calibrator_validates_source_arguments(field_geo):
    field_map = FieldMap(
        np.ones((1, 1, 1, 3), dtype=float),
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )

    with pytest.raises(ValueError, match="exactly one"):
        FieldCalibrator()

    with pytest.raises(ValueError, match="exactly one"):
        FieldCalibrator(map_file="fake.root", field_map=field_map)


def test_field_calibrator_requires_initialized_geometry(monkeypatch):
    field_map = FieldMap(
        np.ones((1, 1, 1, 3), dtype=float),
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )

    def raise_missing_geometry():
        raise ValueError("Geometry singleton instance is not initialized.")

    monkeypatch.setattr(field_module.GeoManager, "get_instance", raise_missing_geometry)

    with pytest.raises(ValueError, match="Geometry singleton"):
        FieldCalibrator(field_map=field_map)


def test_field_calibrator_loads_map_file(monkeypatch):
    field_map = FieldMap(
        np.ones((1, 1, 1, 3), dtype=float),
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )
    calls = []

    def fake_from_root(map_file, map_prefix, bounds="zero"):
        calls.append((map_file, map_prefix, bounds))
        return field_map

    monkeypatch.setattr(FieldMap, "from_root", fake_from_root)
    monkeypatch.setattr(field_module.GeoManager, "get_instance", make_two_tpc_geo)

    calibrator = FieldCalibrator(
        map_file="fake.root",
        map_prefix="TrueBkwd_Displacement",
        bounds="zero",
        num_tpcs=2,
    )

    assert calibrator.field_map is field_map
    assert calls == [("fake.root", "TrueBkwd_Displacement", "zero")]


def test_field_map_loads_root_th3_components(monkeypatch):
    fake_root = FakeROOT()
    monkeypatch.setattr(field_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(field_module, "ROOT", fake_root)

    field_map = FieldMap.from_root("fake.root")

    assert fake_root.opened == ("fake.root", "r")
    assert fake_root.file.closed
    assert np.allclose(field_map.range, [[0.0, 2.0], [-1.0, 1.0], [10.0, 14.0]])
    assert np.all(field_map.bins == [2, 1, 2])
    assert np.allclose(field_map.values[1, 0, 1], [104.0, 204.0, 304.0])


def test_field_map_rejects_bad_root_file(monkeypatch):
    fake_root = FakeROOT()
    fake_root.file = None
    monkeypatch.setattr(field_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(field_module, "ROOT", fake_root)

    with pytest.raises(OSError, match="Could not open"):
        FieldMap.from_root("missing.root")

    fake_root = FakeROOT()
    fake_root.file.zombie = True
    monkeypatch.setattr(field_module, "ROOT", fake_root)

    with pytest.raises(OSError, match="Could not open"):
        FieldMap.from_root("zombie.root")


def test_field_map_rejects_missing_root_histogram(monkeypatch):
    fake_root = FakeROOT()
    del fake_root.file.hists["TrueFwd_Displacement_Z"]
    monkeypatch.setattr(field_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(field_module, "ROOT", fake_root)

    with pytest.raises(KeyError, match="TrueFwd_Displacement_Z"):
        FieldMap.from_root("fake.root")

    assert fake_root.file.closed


def test_field_map_rejects_mismatched_component_geometry(monkeypatch):
    fake_root = FakeROOT()
    fake_root.file.hists["TrueFwd_Displacement_Z"] = FakeTH3(
        290.0, axes=[FakeAxis(0.0, 3.0), FakeAxis(-1.0, 1.0), FakeAxis(10.0, 14.0)]
    )
    monkeypatch.setattr(field_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(field_module, "ROOT", fake_root)

    with pytest.raises(ValueError, match="same binning"):
        FieldMap.from_root("fake.root")

    assert fake_root.file.closed


def test_field_map_requires_root_for_root_loading(monkeypatch):
    monkeypatch.setattr(field_module, "ROOT_AVAILABLE", False)
    with pytest.raises(ImportError, match="ROOT"):
        FieldMap.from_root("fake.root")


class FakeAxis:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def GetXmin(self):
        return self.low

    def GetXmax(self):
        return self.high


class FakeTH3:
    def __init__(self, offset, axes=None):
        self.offset = offset
        self.axes = axes or [
            FakeAxis(0.0, 2.0),
            FakeAxis(-1.0, 1.0),
            FakeAxis(10.0, 14.0),
        ]

    def GetXaxis(self):
        return self.axes[0]

    def GetYaxis(self):
        return self.axes[1]

    def GetZaxis(self):
        return self.axes[2]

    def GetNbinsX(self):
        return 2

    def GetNbinsY(self):
        return 1

    def GetNbinsZ(self):
        return 2

    def GetBinContent(self, ix, iy, iz):
        return self.offset + ix + 10 * iy + iz


class FakeTFile:
    def __init__(self):
        self.closed = False
        self.zombie = False
        self.hists = {
            "TrueFwd_Displacement_X": FakeTH3(90.0),
            "TrueFwd_Displacement_Y": FakeTH3(190.0),
            "TrueFwd_Displacement_Z": FakeTH3(290.0),
        }

    def IsZombie(self):
        return self.zombie

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


def make_two_tpc_geo():
    modules = [
        SimpleNamespace(
            center=np.asarray([-1.0, 0.0, 0.0]),
            dimensions=np.asarray([2.0, 2.0, 2.0]),
            boundaries=np.asarray([[-2.0, 0.0], [-1.0, 1.0], [-1.0, 1.0]]),
        ),
        SimpleNamespace(
            center=np.asarray([1.0, 0.0, 0.0]),
            dimensions=np.asarray([2.0, 2.0, 2.0]),
            boundaries=np.asarray([[0.0, 2.0], [-1.0, 1.0], [-1.0, 1.0]]),
        ),
    ]
    chambers = modules

    def get_closest_tpc(points):
        return np.where(points[:, 0] < 0.0, 0, 1)

    return SimpleNamespace(
        tpc=SimpleNamespace(
            center=np.asarray([0.0, 0.0, 0.0]),
            modules=modules,
            chambers=chambers,
            num_modules=2,
            num_chambers=2,
            num_chambers_per_module=1,
        ),
        get_closest_tpc=get_closest_tpc,
    )
