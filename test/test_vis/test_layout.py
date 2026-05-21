"""Tests for layout visualization helpers."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from spine.vis.layout import (
    apply_latex_style,
    color_rgba,
    dual_figure3d,
    layout3d,
    set_latex_size,
)
from spine.vis.point import scatter_points

POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


class MetaStub:
    lower = np.array([-1.0, -2.0, -3.0])
    upper = np.array([9.0, 8.0, 7.0])
    size = np.array([1.0, 2.0, 5.0])

    def to_px(self, points):
        return points


def test_layout_and_dual_figure_helpers():
    layout = layout3d(ranges=np.array([[0, 1], [0, 2], [0, 3]], dtype=float))
    fig = dual_figure3d([scatter_points(POINTS)[0]], [scatter_points(POINTS)[0]])
    width, height = set_latex_size(250)

    assert layout.scene.xaxis.range == (0, 1)
    assert len(fig.data) == 2
    assert width > height > 0
    assert color_rgba((1, 2, 3), 0.5) == "rgba(1, 2, 3, 0.5)"


def test_layout3d_uses_point_ranges_meta_and_dark_options(monkeypatch):
    point_layout = layout3d(ranges=POINTS, titles=["a", "b", "c"], dark=True)
    meta_layout = layout3d(meta=MetaStub())
    meta_detector_layout = layout3d(meta=MetaStub(), detector_coords=True)
    geo_layout = layout3d(
        geo=SimpleNamespace(get_boundaries=lambda **_: np.array([[0.0, 1.0]] * 3)),
        detector_coords=True,
    )
    monkeypatch.setattr(
        "spine.vis.layout.GeoManager.get_instance",
        lambda: SimpleNamespace(get_boundaries=lambda **_: np.array([[0.0, 1.0]] * 3)),
    )
    managed_geo_layout = layout3d(use_geo=True, detector_coords=True)
    pixel_geo_layout = layout3d(
        geo=SimpleNamespace(get_boundaries=lambda **_: np.array([[0.0, 1.0]] * 3)),
        meta=MetaStub(),
    )

    assert point_layout.paper_bgcolor == "black"
    assert point_layout.scene.xaxis.title.text == "a"
    assert meta_layout.scene.xaxis.range == (0, 10)
    assert meta_layout.scene.yaxis.range == (0, 5)
    assert meta_layout.scene.zaxis.range == (0, 2)
    assert meta_detector_layout.scene.xaxis.range == (-1, 9)
    assert geo_layout.scene.xaxis.range == (-0.1, 1.1)
    assert managed_geo_layout.scene.xaxis.range == (-0.1, 1.1)
    assert pixel_geo_layout.scene.xaxis.range == (-0.1, 1.1)


def test_layout_helpers_cover_style_and_validation(monkeypatch):
    layout = layout3d(width=10, height=20)
    fig = dual_figure3d([], [], layout=layout, width=100, height=50)
    monkeypatch.setattr("spine.vis.layout.go.FigureWidget", lambda fig: fig)
    sync_fig = dual_figure3d(
        [scatter_points(POINTS)[0]],
        [scatter_points(POINTS)[0]],
        synchronize=True,
    )
    camera = {"eye": {"x": 1.0, "y": 2.0, "z": 3.0}}
    sync_fig.layout.scene1.camera = camera
    sync_fig.layout.scene2.camera = {"eye": {"x": 3.0, "y": 2.0, "z": 1.0}}

    with matplotlib.rc_context():
        apply_latex_style()
        assert matplotlib.rcParams["text.usetex"]

    assert fig.layout.width == 100
    assert fig.layout.height == 50
    assert sync_fig.layout.scene1.camera.eye.x == 3.0

    with pytest.raises(ValueError, match="geo"):
        layout3d(ranges=np.array([[0, 1], [0, 1], [0, 1]]), geo=object())
    with pytest.raises(ValueError, match="metadata"):
        layout3d(
            geo=SimpleNamespace(get_boundaries=lambda **_: np.array([[0.0, 1.0]] * 3))
        )
    with pytest.raises(ValueError, match="ranges"):
        layout3d(ranges=np.array([[0, 1], [0, 1], [0, 1]]), meta=MetaStub())
    with pytest.raises(ValueError, match="shape"):
        layout3d(ranges=np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="upper bound"):
        layout3d(ranges=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="one title"):
        layout3d(titles=["x"])
