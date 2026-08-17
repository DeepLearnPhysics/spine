"""Tests for renderer-neutral visualization scenes."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from plotly import graph_objs as go

from spine.constants import TRACK_SHP
from spine.data.out import RecoInteraction, RecoParticle, TruthParticle
from spine.geo import Geometry
from spine.vis import (
    BoxLayer,
    Drawer,
    LineLayer,
    LineStyle,
    MarkerLayer,
    MeshLayer,
    MeshStyle,
    PlotlyBackend,
    PointLayer,
    PointStyle,
    Scene,
    SceneView,
    VectorLayer,
    get_backend,
    plotly_trace_to_layer,
    register_backend,
    render_scene,
)

scene_backend = import_module("spine.vis.scene.backend")
out_layers = import_module("spine.vis.drawer.out.layers")
out_scene = import_module("spine.vis.drawer.out.scene")


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


def test_geometric_layers_normalize_and_validate():
    """Line, vector, mesh and box layers should expose compact buffers."""
    line = LineLayer(np.zeros((2, 2, 3)), values=[1, 2], object_ids=[1, 2])
    vector = VectorLayer(
        np.zeros((2, 3)), np.ones((2, 3)), values=[3, 4], object_ids=[1, 2]
    )
    mesh = MeshLayer(np.zeros((3, 3)), [[0, 1, 2]], values=[1, 2, 3])
    box = BoxLayer(np.asarray([[[0, 0, 0], [1, 2, 3]]]), values=[1], object_ids=[4])

    assert line.segments.dtype == np.float32
    assert line.values.flags.c_contiguous
    assert vector.vectors.flags.c_contiguous
    assert vector.values.tolist() == [3, 4]
    assert mesh.faces.dtype == np.int32
    assert mesh.values.flags.c_contiguous
    assert box.bounds.shape == (1, 2, 3)
    assert box.values.flags.c_contiguous

    with pytest.raises(ValueError, match="segments"):
        LineLayer(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Line object IDs"):
        LineLayer(np.zeros((2, 2, 3)), object_ids=[1])
    with pytest.raises(ValueError, match="one value per segment"):
        LineLayer(np.zeros((2, 2, 3)), values=[1])
    with pytest.raises(ValueError, match="origins"):
        VectorLayer(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="directions"):
        VectorLayer(np.zeros((2, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="Vector object IDs"):
        VectorLayer(np.zeros((2, 3)), np.zeros((2, 3)), object_ids=[1])
    with pytest.raises(ValueError, match="Vector values"):
        VectorLayer(np.zeros((2, 3)), np.zeros((2, 3)), values=[1])
    with pytest.raises(ValueError, match="vertices"):
        MeshLayer(np.zeros((3, 2)), [[0, 1, 2]])
    with pytest.raises(ValueError, match="faces"):
        MeshLayer(np.zeros((3, 3)), [0, 1, 2])
    with pytest.raises(ValueError, match="existing"):
        MeshLayer(np.zeros((3, 3)), [[0, 1, 3]])
    with pytest.raises(ValueError, match="one value per vertex"):
        MeshLayer(np.zeros((3, 3)), [[0, 1, 2]], values=[1, 2])
    with pytest.raises(ValueError, match="bounds"):
        BoxLayer(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="upper"):
        BoxLayer(np.asarray([[[1, 0, 0], [0, 1, 1]]]))
    with pytest.raises(ValueError, match="Box object IDs"):
        BoxLayer(np.zeros((2, 2, 3)), object_ids=[1])
    with pytest.raises(ValueError, match="one value per box"):
        BoxLayer(np.zeros((2, 2, 3)), values=[1])


def test_plotly_adapter_supports_all_3d_trace_types():
    """Established Plotly glyphs should map onto neutral primitives."""
    marker = go.Scatter3d(
        x=[0], y=[1], z=[2], mode="markers", marker={"size": 5, "symbol": "diamond"}
    )
    line = go.Scatter3d(
        x=[0, 1, None, 2, 3],
        y=[0, 1, None, 2, 3],
        z=[0, 1, None, 2, 3],
        mode="lines",
        line={"width": 4, "color": "red"},
    )
    valued_line = go.Scatter3d(
        x=[0, 1, None, 2, 3],
        y=[0, 1, None, 2, 3],
        z=[0, 1, None, 2, 3],
        mode="lines",
        line={"color": [0, 2, 0, 4, 6]},
    )
    mesh = go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2])
    hull = go.Mesh3d(
        x=[0, 1, 0, 0],
        y=[0, 0, 1, 0],
        z=[0, 0, 0, 1],
        alphahull=0,
    )
    cone = go.Cone(
        x=[0], y=[0], z=[0], u=[1], v=[0], w=[0], colorscale=[[0, "blue"], [1, "blue"]]
    )

    assert isinstance(plotly_trace_to_layer(marker), MarkerLayer)
    assert plotly_trace_to_layer(marker).symbol == "diamond"
    assert plotly_trace_to_layer(line).segments.shape == (2, 2, 3)
    assert plotly_trace_to_layer(valued_line).values.tolist() == [1.0, 5.0]
    assert isinstance(plotly_trace_to_layer(mesh), MeshLayer)
    assert plotly_trace_to_layer(hull).faces.shape == (4, 3)
    assert isinstance(plotly_trace_to_layer(cone), VectorLayer)

    with pytest.raises(TypeError, match="mode"):
        plotly_trace_to_layer(go.Scatter3d(x=[0], y=[0], z=[0], mode="text"))
    with pytest.raises(TypeError, match="trace type"):
        plotly_trace_to_layer(go.Bar(x=[0], y=[1]))
    with pytest.raises(ValueError, match="convex"):
        plotly_trace_to_layer(
            go.Mesh3d(
                x=[0, 1, 0, 0],
                y=[0, 0, 1, 0],
                z=[0, 0, 0, 1],
                alphahull=2,
            )
        )
    with pytest.raises(ValueError, match="matching lengths"):
        plotly_trace_to_layer(
            go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], k=[2])
        )
    with pytest.raises(ValueError, match="line colors"):
        plotly_trace_to_layer(
            go.Scatter3d(
                x=[0, 1], y=[0, 1], z=[0, 1], mode="lines", line={"color": [1]}
            )
        )
    assert plotly_trace_to_layer(
        go.Mesh3d(x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0])
    ).faces.shape == (0, 3)
    with pytest.raises(ValueError, match="three-dimensional"):
        plotly_trace_to_layer(go.Mesh3d(x=[0, 1, 0, 1], y=[0, 0, 1, 1], z=[0, 0, 0, 0]))


def test_plotly_backend_renders_all_neutral_layers():
    """The compatibility backend should render every neutral primitive."""
    layers = [
        MarkerLayer(np.zeros((1, 3)), symbol="diamond"),
        LineLayer(np.zeros((1, 2, 3)), values=[1], style=LineStyle(color="red")),
        VectorLayer(np.zeros((1, 3)), np.ones((1, 3)), style=LineStyle(color="blue")),
        VectorLayer(
            np.zeros((1, 3)),
            np.ones((1, 3)),
            values=[1],
            style=LineStyle(colorscale="Viridis", cmin=0, cmax=1),
        ),
        MeshLayer(
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            [[0, 1, 2]],
            values=2,
            style=MeshStyle(color="green"),
        ),
        MeshLayer(
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            [[0, 1, 2]],
            style=MeshStyle(color="blue", wireframe=True),
        ),
        BoxLayer(np.asarray([[[0, 0, 0], [1, 1, 1]]]), values=[2]),
        BoxLayer(
            np.asarray([[[0, 0, 0], [1, 1, 1]]]),
            values=[3],
            draw_faces=True,
            mesh_style=MeshStyle(color="orange"),
        ),
    ]
    figure = Scene([SceneView("all", layers)]).render("plotly")

    assert [trace.type for trace in figure.data] == [
        "scatter3d",
        "scatter3d",
        "cone",
        "scatter3d",
        "cone",
        "mesh3d",
        "scatter3d",
        "scatter3d",
        "mesh3d",
    ]
    with pytest.raises(TypeError, match="Unsupported scene layer"):
        Scene([SceneView("bad", [object()])]).render("plotly")
    invalid_line = LineLayer(np.zeros((1, 2, 3)))
    invalid_line.values = np.asarray([1, 2])
    with pytest.raises(ValueError, match="one scalar per segment"):
        PlotlyBackend._line_trace(invalid_line)
    assert len(figure.data[7].line.color) == 36
    assert len(figure.data[8].intensity) == 8


def test_plotly_backend_uses_scene_layout_metadata():
    """Plotly rendering should preserve neutral detector layout context."""
    scene = Scene(
        [SceneView("all", [PointLayer(np.zeros((1, 3)))])],
        metadata={
            "bounds": [[0.0, 10.0], [0.0, 20.0], [0.0, 30.0]],
            "detector_coords": True,
            "up_dir": [1.0, 0.0, 0.0],
        },
    )

    figure = scene.render("plotly")

    assert figure.layout.scene.xaxis.range == (0, 10)
    assert figure.layout.scene.yaxis.range == (0, 20)
    assert figure.layout.scene.zaxis.range == (0, 30)
    assert figure.layout.scene.xaxis.title.text == "x [cm]"
    assert figure.layout.scene.camera.up.to_plotly_json() == {
        "x": 1.0,
        "y": 0.0,
        "z": 0.0,
    }

    override = scene.render(
        "plotly",
        ranges=np.asarray([[1.0, 2.0]] * 3),
        detector_coords=False,
    )
    assert override.layout.scene.xaxis.range == (1, 2)
    assert override.layout.scene.xaxis.title.text == "x [pixel]"


def test_drawer_scene_plotly_layout_matches_direct_drawer():
    """The neutral Plotly path should retain the direct drawer layout."""
    geo = Geometry(
        name="test",
        tag="test",
        version="1",
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[5.0, 10.0, 15.0]],
            "module_ids": [0],
            "drift_dirs": [[1.0, 0.0, 0.0]],
        },
        up_dir=[1.0, 0.0, 0.0],
    )
    particle = RecoParticle(id=0, index=np.array([0], dtype=np.int32))
    drawer = Drawer(
        {
            "points": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            "reco_particles": [particle],
        },
        draw_mode="reco",
        geo=geo,
    )

    direct = drawer.get("particles")
    scene = drawer.get_scene("particles")
    compatible = scene.render("plotly")

    for axis in ("xaxis", "yaxis", "zaxis"):
        direct_axis = getattr(direct.layout.scene, axis)
        compatible_axis = getattr(compatible.layout.scene, axis)
        assert compatible_axis.range == direct_axis.range
        assert compatible_axis.title.text == direct_axis.title.text
    assert compatible.layout.scene.camera == direct.layout.scene.camera
    assert scene.metadata["bounds"] == [
        [0.0, 10.0],
        [0.0, 20.0],
        [0.0, 30.0],
    ]
    assert scene.metadata["layout_bounds"] == [
        [-1.0, 11.0],
        [-2.0, 22.0],
        [-3.0, 33.0],
    ]


def test_plotly_backend_expands_line_hovertext():
    """One line label should be attached to both vertices of its segment."""
    segments = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    trace = PlotlyBackend._line_trace(
        LineLayer(segments, hovertext=["first", "second"])
    )

    assert trace["hovertext"].tolist() == [
        "first",
        "first",
        None,
        "second",
        "second",
        None,
    ]
    scalar = PlotlyBackend._line_trace(LineLayer(segments, hovertext="shared"))
    assert scalar["hovertext"].tolist() == [
        "shared",
        "shared",
        None,
        "shared",
        "shared",
        None,
    ]
    array_scalar = PlotlyBackend._line_trace(
        LineLayer(segments, hovertext=np.asarray("shared"))
    )
    assert array_scalar["hovertext"].tolist() == scalar["hovertext"].tolist()

    with pytest.raises(ValueError, match="one label per segment"):
        PlotlyBackend._line_trace(LineLayer(segments, hovertext=["only one"]))


def test_plotly_backend_expands_box_hovertext():
    """Wireframe boxes should repeat each box label across all of its edges."""
    boxes = BoxLayer(
        np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            ]
        ),
        hovertext=["first", "second"],
    )

    trace = PlotlyBackend._box_trace(boxes)

    assert trace["hovertext"][:36].tolist().count("first") == 24
    assert trace["hovertext"][36:].tolist().count("second") == 24

    boxes.draw_faces = True
    face_trace = PlotlyBackend._box_trace(boxes)
    assert face_trace["hovertext"].tolist() == ["first"] * 8 + ["second"] * 8


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

    drawer = Drawer(data, draw_mode="both", split_scene=True)
    drawer.geo = SimpleNamespace(
        get_boundaries=lambda **kwargs: np.asarray([[0, 1], [0, 1], [0, 1]])
    )
    drawer.geo_drawer = SimpleNamespace(
        tpc_traces=lambda **kwargs: [
            go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="lines")
        ]
    )
    scene = drawer.get_scene("particles", draw_raw=True)

    assert len(scene.views) == 2
    assert [len(view.layers) for view in scene.views] == [3, 3]
    assert scene.views[0].layers[0].style.cmax == 4.0
    assert scene.views[1].layers[0].style.cmax == 1.0
    assert scene.views[1].layers[1].point_count == 0


def test_drawer_scene_builds_native_auxiliary_layers():
    """Full scenes should expose raw, marker and vector layers natively."""
    track = RecoParticle(
        id=7,
        index=np.array([0, 1], dtype=np.int32),
        shape=TRACK_SHP,
        start_point=np.zeros(3, dtype=np.float32),
        end_point=np.ones(3, dtype=np.float32),
        start_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    shower = RecoParticle(
        id=8,
        index=np.array([2], dtype=np.int32),
        shape=0,
        start_point=np.ones(3, dtype=np.float32),
        end_point=np.ones(3, dtype=np.float32) * 2,
        start_dir=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    data = {
        "points": np.arange(9, dtype=np.float32).reshape(3, 3),
        "depositions": np.arange(1, 4, dtype=np.float32),
        "reco_particles": [track, shower],
    }

    scene = Drawer(data, draw_mode="reco").get_scene(
        "particles",
        draw_raw=True,
        draw_end_points=True,
        draw_directions=True,
    )

    assert [type(layer) for layer in scene.views[0].layers] == [
        PointLayer,
        PointLayer,
        MarkerLayer,
        MarkerLayer,
        VectorLayer,
    ]
    assert scene.views[0].layers[3].object_ids.tolist() == [0]
    assert scene.views[0].layers[4].object_ids.tolist() == [0, 1]
    assert scene.views[0].layers[2].values.tolist() == [0, 1]
    assert scene.views[0].layers[3].values.tolist() == [0]
    assert scene.views[0].layers[4].values.tolist() == [0, 1]

    legacy_scene = Drawer(data, draw_mode="reco", match_aux_colors=False).get_scene(
        "particles", draw_end_points=True, draw_directions=True
    )
    assert legacy_scene.views[0].layers[1].values == "black"
    assert legacy_scene.views[0].layers[3].values is None
    assert legacy_scene.views[0].layers[3].style.color == "black"


def test_drawer_scene_builds_vertex_and_meta_bounds():
    """Interaction vertices and metadata bounds should remain renderer neutral."""
    interaction = RecoInteraction(
        id=0,
        index=np.array([0], dtype=np.int32),
        vertex=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    data = {
        "points": np.ones((1, 3), dtype=np.float32),
        "reco_interactions": [interaction],
        "meta": SimpleNamespace(
            lower=np.zeros(3), upper=np.ones(3), size=np.full(3, 0.25)
        ),
    }

    scene = Drawer(data, draw_mode="reco").get_scene("interactions", draw_vertices=True)

    assert isinstance(scene.views[0].layers[-1], MarkerLayer)
    assert scene.views[0].layers[-1].symbol == "diamond"
    assert scene.views[0].layers[-1].values.tolist() == [0]
    assert scene.metadata["bounds"] == [[0.0, 1.0]] * 3

    pixel_scene = Drawer(data, draw_mode="reco", detector_coords=False).get_scene(
        "interactions"
    )
    assert pixel_scene.metadata["bounds"] == [[0.0, 4.0]] * 3

    legacy_scene = Drawer(data, draw_mode="reco", match_aux_colors=False).get_scene(
        "interactions", draw_vertices=True
    )
    assert legacy_scene.views[0].layers[-1].values == "green"


def test_drawer_scene_adapts_lite_objects():
    """Lite object glyphs should be accepted at the neutral scene boundary."""
    particle = RecoParticle(
        id=0,
        index=np.array([0], dtype=np.int32),
        shape=TRACK_SHP,
        start_point=np.zeros(3),
        end_point=np.ones(3),
        start_dir=np.array([1.0, 0.0, 0.0]),
    )
    drawer = Drawer({"reco_particles": [particle]}, draw_mode="reco", lite=True)

    scene = drawer.get_scene("particles")

    assert scene.views[0].layers
    assert all(
        isinstance(layer, (MarkerLayer, MeshLayer, LineLayer))
        for layer in scene.views[0].layers
    )
    with pytest.raises(RuntimeError, match="raw input"):
        drawer.get_scene("particles", draw_raw=True)


def test_drawer_scene_adapts_detector_optical_and_crt(monkeypatch):
    """Every detector-specific drawing path should produce neutral layers."""
    interaction = RecoInteraction(
        id=0,
        index=np.array([0], dtype=np.int32),
        vertex=np.zeros(3, dtype=np.float32),
    )
    drawer = Drawer(
        {
            "points": np.zeros((1, 3), dtype=np.float32),
            "reco_interactions": [interaction],
            "flashes": [object()],
            "flash_hypotheses": [object()],
            "crthits": [object()],
        },
        draw_mode="reco",
        split_scene=False,
    )
    calls = []

    class FakeGeometry:
        up_dir = np.array([1.0, 0.0, 0.0])

        def get_boundaries(self, *, with_optical, with_crt):
            calls.append((with_optical, with_crt))
            return np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    drawer.geo = FakeGeometry()
    drawer.geo_drawer = SimpleNamespace(
        tpc_traces=lambda **kwargs: [
            go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode="lines")
        ]
    )
    monkeypatch.setattr(out_scene, "get_flash_pe", lambda *args, **kwargs: np.ones(2))
    monkeypatch.setattr(
        out_scene,
        "get_flash_hypothesis_pe",
        lambda *args, **kwargs: np.ones(2) * 2,
    )
    monkeypatch.setattr(
        out_scene,
        "build_flash_trace",
        lambda **kwargs: [
            go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="flash")
        ],
    )
    monkeypatch.setattr(
        out_scene,
        "build_flash_hypothesis_trace",
        lambda **kwargs: [
            go.Mesh3d(
                x=[0, 1, 0],
                y=[0, 0, 1],
                z=[0, 0, 0],
                i=[0],
                j=[1],
                k=[2],
                name="hypothesis",
            )
        ],
    )
    monkeypatch.setattr(
        out_scene,
        "build_crt_trace",
        lambda **kwargs: [
            go.Scatter3d(x=[0, 1], y=[0, 1], z=[1, 1], mode="lines", name="crt")
        ],
    )

    scene = drawer.get_scene(
        "interactions",
        draw_flashes=True,
        draw_flash_hypotheses=True,
        draw_crthits=True,
        optical_size_by_pe=True,
    )

    kinds = [layer.metadata.get("kind") for layer in scene.views[0].layers]
    assert kinds == [None, "optical", "optical", "crt", "geometry"]
    assert calls == [(True, True)]
    assert scene.metadata["bounds"] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert scene.metadata["up_dir"] == [1.0, 0.0, 0.0]


def test_drawer_scene_skips_empty_truth_auxiliary_glyphs():
    """Empty truth objects should not contribute markers or vectors."""
    particle = TruthParticle(
        id=0,
        index=np.empty(0, dtype=np.int32),
        index_adapt=np.empty(0, dtype=np.int32),
        shape=TRACK_SHP,
        start_point=np.zeros(3, dtype=np.float32),
        end_point=np.ones(3, dtype=np.float32),
        momentum=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    drawer = Drawer(
        {
            "points_label": np.empty((0, 3), dtype=np.float32),
            "truth_particles": [particle],
        },
        draw_mode="truth",
    )

    scene = drawer.get_scene("particles", draw_end_points=True, draw_directions=True)

    assert [layer.point_count for layer in scene.views[0].layers[1:3]] == [0, 0]
    assert len(scene.views[0].layers[3].origins) == 0


def test_drawer_scene_validates_all_auxiliary_requests():
    """Neutral scenes should preserve every established request failure."""
    points = np.empty((0, 3), dtype=np.float32)
    particle_drawer = Drawer({"points": points, "reco_particles": []}, draw_mode="reco")
    interaction_drawer = Drawer(
        {"points": points, "reco_interactions": []}, draw_mode="reco"
    )

    with pytest.raises(ValueError, match="Interactions do not have end"):
        interaction_drawer.get_scene("interactions", draw_end_points=True)
    with pytest.raises(ValueError, match="Interactions do not have direction"):
        interaction_drawer.get_scene("interactions", draw_directions=True)
    with pytest.raises(ValueError, match="provide interactions"):
        particle_drawer.get_scene("particles", draw_vertices=True)
    with pytest.raises(ValueError, match="`flashes`"):
        interaction_drawer.get_scene("interactions", draw_flashes=True)
    with pytest.raises(ValueError, match="`flash_hypotheses`"):
        interaction_drawer.get_scene("interactions", draw_flash_hypotheses=True)
    with pytest.raises(ValueError, match="interactions"):
        particle_drawer.data["flashes"] = []
        particle_drawer.get_scene("particles", draw_flashes=True)
    with pytest.raises(ValueError, match="`crthits`"):
        particle_drawer.get_scene("particles", draw_crthits=True)


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


def test_drawer_scene_skips_nonportable_color_mappings(monkeypatch):
    """Client recoloring should retain only aligned scalar numeric mappings."""
    particle = RecoParticle(
        id=1,
        index=np.array([0, 1], dtype=np.int32),
        shape=TRACK_SHP,
        pid=2,
    )
    data = {
        "points": np.ones((2, 3), dtype=np.float32),
        "depositions": np.array([1.0, 2.0], dtype=np.float32),
        "reco_particles": [particle],
    }
    original = out_layers.build_object_colors

    def build_colors(**kwargs):
        result = original(**kwargs)
        if kwargs["color_attr"] == "pid":
            result["color"] = np.ones((2, 2), dtype=np.float32)
        elif kwargs["color_attr"] == "shape":
            result["color"] = np.asarray(["track", "track"])
        return result

    monkeypatch.setattr(out_layers, "build_object_colors", build_colors)
    scene = Drawer(data, draw_mode="reco").get_scene("particles", attr=["pid", "shape"])

    assert "pid" not in scene.views[0].layers[0].metadata["attribute_styles"]
    assert "shape" not in scene.views[0].layers[0].metadata["attribute_styles"]


def test_drawer_scene_identifies_pointwise_hover_attributes():
    """Scene metadata should distinguish pointwise from object attributes."""
    particle = RecoParticle(
        id=1,
        index=np.array([0, 1], dtype=np.int32),
        shape=TRACK_SHP,
        is_contained=True,
        depositions=np.array([1.0, 2.0], dtype=np.float32),
    )
    data = {
        "points": np.ones((2, 3), dtype=np.float32),
        "depositions": np.array([1.0, 2.0], dtype=np.float32),
        "reco_particles": [particle],
    }

    scene = Drawer(data, draw_mode="reco").get_scene(
        "particles", attr=["depositions", "is_contained"]
    )

    assert scene.views[0].layers[0].metadata["long_form_attributes"] == ["depositions"]


def test_expand_object_values_covers_alignment_modes():
    """Object values should normalize from every supported alignment mode."""
    expand = out_layers.SceneLayerBuilder.expand_object_values
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
