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


def test_geo_drawer_requires_meta_for_pixel_coordinates(full_geo):
    """Pixel-coordinate drawing should require metadata conversion context."""
    drawer = GeoDrawer(full_geo, detector_coords=False)

    with pytest.raises(AssertionError, match="meta"):
        drawer.tpc_traces()

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
