"""Tests for renderer-neutral visualization scenes."""

import numpy as np
import pytest

from spine.constants import TRACK_SHP
from spine.data.out import RecoParticle
from spine.vis import Drawer, PointLayer, Scene, SceneView, register_backend


def test_point_layer_normalizes_gpu_arrays():
    """Point layers should expose contiguous, compact GPU-ready arrays."""
    layer = PointLayer(
        positions=np.arange(12, dtype=np.float64).reshape(4, 3),
        object_ids=[2, 2, 5, 5],
        object_offsets=[0, 2, 4],
        attributes={"energy": np.arange(4, dtype=np.float32)},
    )

    assert layer.positions.dtype == np.float32
    assert layer.positions.flags.c_contiguous
    assert layer.object_ids.dtype == np.int32
    assert layer.point_count == 4
    assert layer.object_count == 2


def test_point_layer_validates_shapes():
    """Malformed point and object arrays should fail at the scene boundary."""
    with pytest.raises(ValueError, match="shape"):
        PointLayer(np.empty((3, 2)))
    with pytest.raises(ValueError, match="object IDs"):
        PointLayer(np.empty((3, 3)), object_ids=np.arange(2))
    with pytest.raises(ValueError, match="offsets"):
        PointLayer(np.empty((3, 3)), object_offsets=[0, 2])


def test_scene_accepts_custom_backend():
    """Third-party renderers should plug in without depending on Plotly."""

    class CountBackend:
        def render(self, scene, **kwargs):
            return sum(
                layer.point_count for view in scene.views for layer in view.layers
            )

    register_backend("test-count", CountBackend, replace=True)
    scene = Scene([SceneView("event", [PointLayer(np.zeros((3, 3)))])])

    assert scene.render("test-count") == 3
    assert scene.render(CountBackend()) == 3


def test_drawer_builds_scene_and_plotly_backend():
    """Drawer scenes should preserve object membership for either trace strategy."""
    particles = [
        RecoParticle(
            id=7,
            index=np.array([0, 1], dtype=np.int32),
            shape=TRACK_SHP,
            pid=2,
        ),
        RecoParticle(
            id=9,
            index=np.array([2], dtype=np.int32),
            shape=TRACK_SHP,
            pid=3,
        ),
    ]
    data = {
        "points": np.arange(9, dtype=np.float32).reshape(3, 3),
        "reco_particles": particles,
    }

    scene = Drawer(data, draw_mode="reco").get_scene(
        "particles", attr=["pid"], color_attr="pid"
    )
    layer = scene.views[0].layers[0]

    assert layer.object_ids.tolist() == [7, 7, 9]
    assert layer.object_offsets.tolist() == [0, 2, 3]
    assert layer.attributes["pid"].tolist() == [2, 2, 3]
    assert len(scene.render("plotly").data) == 1
    split_figure = scene.render("plotly", split_objects=True)
    assert len(split_figure.data) == 2
    assert [len(trace.x) for trace in split_figure.data] == [2, 1]
