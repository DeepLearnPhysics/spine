"""Tests for renderer-neutral visualization scenes."""

from importlib import import_module

import numpy as np
import pytest
from plotly import graph_objs as go

from spine.constants import TRACK_SHP
from spine.data.out import RecoParticle, TruthParticle
from spine.vis import (
    Drawer,
    PlotlyBackend,
    PointLayer,
    PointStyle,
    Scene,
    SceneView,
    get_backend,
    register_backend,
    render_scene,
)

scene_backend = import_module("spine.vis.scene.backend")


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
        PointLayer(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="object IDs"):
        PointLayer(np.zeros((3, 3)), object_ids=np.arange(2))
    with pytest.raises(ValueError, match="offsets"):
        PointLayer(np.zeros((3, 3)), object_offsets=[0, 2])
    with pytest.raises(ValueError, match="non-empty"):
        PointLayer(np.empty((0, 3)), object_offsets=[])
    with pytest.raises(ValueError, match="attribute"):
        PointLayer(np.zeros((3, 3)), attributes={"energy": np.arange(2)})
    with pytest.raises(ValueError, match="values"):
        PointLayer(np.zeros((3, 3)), values=np.arange(2))

    layer = PointLayer(np.empty((0, 3)))
    assert layer.object_count == 0


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
    assert render_scene(scene, "test-count") == 3
    assert render_scene(scene, CountBackend()) == 3

    with pytest.raises(ValueError, match="empty"):
        register_backend("", CountBackend)
    with pytest.raises(ValueError, match="already registered"):
        register_backend("test-count", CountBackend)
    with pytest.raises(ValueError, match="Unknown scene backend"):
        get_backend("missing-test-backend")
    with pytest.raises(TypeError, match="render"):
        scene.render(object())


def test_plotly_backend_validates_and_renders_views():
    """Plotly backend should support empty, single, and dual-view scenes."""
    backend = PlotlyBackend()
    layout = go.Layout(showlegend=False)

    empty_figure = backend.render(Scene(), layout=layout)
    assert empty_figure.layout.showlegend is False

    unnamed = PointLayer(
        np.zeros((2, 3), dtype=np.float32),
        values=1.0,
        hovertext="shared",
        object_offsets=[0, 1, 2],
        style=PointStyle(size=4.0),
    )
    split_figure = Scene([SceneView("single", [unnamed])]).render(
        "plotly", split_objects=True
    )
    assert [trace.name for trace in split_figure.data] == ["0", "1"]

    dual_scene = Scene(
        [
            SceneView("Reco", [PointLayer(np.zeros((1, 3)))]),
            SceneView("Truth", [PointLayer(np.ones((1, 3)))]),
        ]
    )
    dual_figure = backend.render(dual_scene, synchronize=False)
    assert len(dual_figure.data) == 2

    invalid_scene = Scene([SceneView(str(i)) for i in range(3)])
    with pytest.raises(ValueError, match="at most two"):
        backend.render(invalid_scene)


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


def test_drawer_builds_raw_split_scene():
    """Drawer scenes should support raw layers and split truth/reco views."""
    reco = RecoParticle(
        id=2,
        index=np.array([0], dtype=np.int32),
        shape=TRACK_SHP,
    )
    truth = TruthParticle(
        id=3,
        index=np.empty(0, dtype=np.int32),
        index_adapt=np.empty(0, dtype=np.int32),
    )
    data = {
        "points": np.ones((1, 3), dtype=np.float32),
        "depositions": np.array([2.0], dtype=np.float32),
        "points_label": np.empty((0, 3), dtype=np.float32),
        "depositions_label": np.empty(0, dtype=np.float32),
        "reco_particles": [reco],
        "truth_particles": [truth],
    }

    scene = Drawer(data, draw_mode="both", split_scene=True).get_scene(
        "particles", draw_raw=True
    )

    assert len(scene.views) == 2
    assert [len(view.layers) for view in scene.views] == [2, 2]
    assert scene.views[0].layers[0].style.cmax == 4.0
    assert scene.views[1].layers[0].style.cmax == 1.0
    assert scene.views[1].layers[1].point_count == 0


def test_drawer_scene_validates_missing_products():
    """Scene construction should reject absent collections and point arrays."""
    points = np.empty((0, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="reco_particles"):
        Drawer({"points": points}, draw_mode="reco").get_scene("particles")
    with pytest.raises(ValueError, match="points"):
        Drawer({"reco_particles": [RecoParticle()]}, draw_mode="reco").get_scene(
            "particles"
        )
    with pytest.raises(ValueError, match="depositions"):
        Drawer({"points": points, "reco_particles": []}, draw_mode="reco").get_scene(
            "particles", draw_raw=True
        )


def test_drawer_scene_skips_unaligned_numeric_attributes():
    """Unaligned optional attributes should stay out of the neutral layer."""
    particle = RecoParticle(
        id=1,
        index=np.arange(4, dtype=np.int32),
        shape=TRACK_SHP,
        start_point=np.ones(3, dtype=np.float32),
    )
    data = {
        "points": np.ones((4, 3), dtype=np.float32),
        "reco_particles": [particle],
    }

    scene = Drawer(data, draw_mode="reco").get_scene(
        "particles", attr=["id", "start_point"]
    )

    assert "start_point" not in scene.views[0].layers[0].attributes


def test_expand_object_values_covers_alignment_modes():
    """Object values should normalize from every supported alignment mode."""
    expand = Drawer._expand_object_values
    indices = [np.array([0, 2]), np.array([], dtype=np.int64), np.array([1])]

    assert expand(None, indices, 4) is None
    assert expand(3, indices, 4) == 3
    assert expand(np.arange(4), indices, 4).tolist() == [0, 2, 1]
    assert expand([5, 6, 7], indices, 4).tolist() == [5, 5, 7]
    assert expand([np.array([8, 9]), [], np.array([10])], indices, 4).tolist() == [
        8,
        9,
        10,
    ]

    with pytest.raises(ValueError, match="shorter"):
        expand([np.array([1]), [], np.array([2])], indices, 4)
    with pytest.raises(ValueError, match="scalar"):
        expand([1, 2], indices, 4)

    empty = expand([1], [np.array([], dtype=np.int64)], 0, dtype=object)
    assert empty.dtype == object
    assert len(empty) == 0
