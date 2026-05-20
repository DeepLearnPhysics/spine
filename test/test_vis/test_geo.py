"""Tests for geometry visualization helpers."""

import numpy as np
import pytest

from spine.geo.base import Geometry
from spine.vis.geo import GeoDrawer


class MetaStub:
    """Minimal metadata object for coordinate conversion tests."""

    size = np.array([1.0, 1.0, 1.0])

    def to_px(self, points, floor=False):
        """Pretend detector coordinates are already pixel coordinates."""
        return np.floor(points).astype(int) if floor else points


@pytest.fixture(name="full_geo")
def fixture_full_geo():
    """Build a small geometry with TPC, optical, and CRT components."""
    return Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
        optical={
            "volume": "module",
            "shape": "box",
            "dimensions": [2.0, 2.0, 2.0],
            "positions": [[0.0, 15.0, 0.0]],
        },
        crt={
            "dimensions": [[2.0, 2.0, 2.0]],
            "positions": [[0.0, 30.0, 0.0]],
            "normals": [1],
        },
    )


@pytest.fixture(name="multi_shape_geo")
def fixture_multi_shape_geo():
    """Build geometry with mixed optical detector shapes."""
    return Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
        optical={
            "volume": "module",
            "shape": ["box", "ellipsoid", "disk"],
            "dimensions": [[2.0, 2.0, 2.0], [1.0, 2.0, 3.0], [4.0, 4.0, 0.5]],
            "positions": [
                [0.0, 15.0, 0.0],
                [1.0, 15.0, 0.0],
                [2.0, 15.0, 0.0],
            ],
            "shape_ids": [0, 1, 2],
        },
        crt={
            "dimensions": [[2.0, 2.0, 2.0]],
            "positions": [[0.0, 30.0, 0.0]],
            "normals": [1],
        },
    )


def test_geo_drawer_builds_detector_traces(full_geo):
    """GeoDrawer should build TPC, optical, CRT, and aggregate traces."""
    drawer = GeoDrawer(full_geo)

    tpcs = drawer.tpc_traces()
    optical = drawer.optical_traces(color=np.array([2.0]))
    optical_zero = drawer.optical_traces(color=np.array([0.0]), zero_supress=True)
    crt = drawer.crt_traces(draw_ids=[0])
    all_traces = drawer.traces()

    assert len(tpcs) == 2
    assert len(optical) == 1
    assert optical[0].intensity.tolist() == [2.0] * 8
    assert optical_zero == []
    assert len(crt) == 1
    assert len(all_traces) == 4


def test_geo_drawer_rejects_missing_or_bad_components(full_geo):
    """GeoDrawer should reject missing geometry components and bad inputs."""
    no_optical = Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
    )
    drawer = GeoDrawer(no_optical)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("spine.vis.geo.GeoManager.get_instance", lambda: full_geo)
        assert GeoDrawer().geo is full_geo
    with pytest.raises(TypeError, match="Geometry"):
        GeoDrawer(object())
    with pytest.raises(RuntimeError, match="optical"):
        drawer.optical_traces()
    with pytest.raises(RuntimeError, match="CRT"):
        drawer.crt_traces()
    with pytest.raises(ValueError, match="one value"):
        GeoDrawer(full_geo).optical_traces(color=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="hovertext"):
        GeoDrawer(full_geo).optical_traces(hovertext=["a", "b"])


def test_geo_drawer_mixed_optical_shapes_and_legends(multi_shape_geo):
    """Mixed optical shapes should draw boxes, ellipsoids, and disks."""
    drawer = GeoDrawer(multi_shape_geo)

    traces = drawer.optical_traces(
        color=[1.0, 2.0, 0.0],
        hovertext="pd",
        zero_supress=True,
        shared_legend=False,
    )
    volume_traces = drawer.optical_traces(volume_id=0, color=np.array([1.0, 2.0, 3.0]))

    assert len(traces) == 2
    assert traces[0].name == "Optical 0"
    assert {trace.type for trace in volume_traces} == {"mesh3d"}


def test_geo_drawer_mixed_optical_rejects_unknown_shape(multi_shape_geo):
    """Unknown optical detector shapes should be rejected explicitly."""
    multi_shape_geo.optical.volumes[0].shape[0] = "bad"

    with pytest.raises(ValueError, match="not recognized"):
        GeoDrawer(multi_shape_geo).optical_traces()


def test_geo_drawer_requires_meta_for_pixel_coordinates(full_geo):
    """Pixel-coordinate drawing should require metadata conversion context."""
    drawer = GeoDrawer(full_geo, detector_coords=False)

    with pytest.raises(ValueError, match="meta"):
        drawer.tpc_traces()
    with pytest.raises(ValueError, match="meta"):
        drawer.optical_traces()
    with pytest.raises(ValueError, match="meta"):
        drawer.crt_traces()

    traces = drawer.traces(meta=MetaStub())

    assert len(traces) == 4


def test_geo_drawer_show_uses_plotly_show(full_geo, monkeypatch):
    """The show helper should build and display a figure."""
    calls = []
    drawer = GeoDrawer(full_geo)
    monkeypatch.setattr(
        "plotly.graph_objs.Figure.show", lambda self: calls.append(self)
    )

    drawer.show()

    assert len(calls) == 1
