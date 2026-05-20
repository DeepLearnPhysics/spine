"""Tests for public visualization helper functions."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from spine.constants import PART_COL, TRACK_SHP
from spine.data import Particle
from spine.vis.arrow import scatter_arrows
from spine.vis.box import box_trace, box_traces, scatter_boxes
from spine.vis.cluster import scatter_clusters
from spine.vis.cone import cone_trace
from spine.vis.cylinder import cylinder_trace, cylinder_traces
from spine.vis.ellipsoid import ellipsoid_trace, ellipsoid_traces
from spine.vis.evaluation import annotate_heatmap, heatmap
from spine.vis.hull import hull_trace
from spine.vis.layout import (
    apply_latex_style,
    color_rgba,
    dual_figure3d,
    layout3d,
    set_latex_size,
)
from spine.vis.lite import (
    legend_trace,
    scatter_lite,
    scatter_lite_interactions,
    scatter_lite_particles,
    track_line_trace,
)
from spine.vis.network import network_schematic, network_topology
from spine.vis.particle import scatter_particles
from spine.vis.point import scatter_points
from spine.vis.train import TrainDrawer

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
CLUSTS = [
    np.array([0, 1, 2], dtype=np.int64),
    np.array([3, 4], dtype=np.int64),
]


def test_scatter_points_supports_2d_and_rejects_bad_dimension():
    """Point drawing should infer 2D traces and validate requested dimensions."""
    trace = scatter_points(POINTS[:, :2], color=np.array([1.0, 2.0]))[0]

    assert trace.type == "scatter"
    assert trace.marker.color.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="dimension 2 or 3"):
        scatter_points(POINTS, dim=4)


def test_scatter_arrows_builds_trunks_and_tips():
    """Arrow drawing should produce a line trace plus a cone-tip trace."""
    traces = scatter_arrows(
        POINTS[:2],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        color=["red", "blue"],
        hovertext=["a", "b"],
    )

    assert len(traces) == 2
    assert traces[0].mode == "lines"
    assert traces[1].type == "cone"
    assert len(traces[1].x) == 2


def test_scatter_arrows_handles_scalar_hovertext():
    """Arrow drawing should include scalar hovertext in the template."""
    traces = scatter_arrows(
        POINTS[:1],
        np.array([[1.0, 0.0, 0.0]]),
        hovertext="direction",
    )

    assert "direction" in traces[0].text[0]


def test_box_helpers_draw_edges_faces_and_scatter_boxes():
    """Box helpers should support edge, face, list, and coordinate APIs."""
    edge_trace = box_trace(np.zeros(3), np.ones(3), color="black")
    edge_with_text = box_trace(np.zeros(3), np.ones(3), hovertext=["edge"] * 8)
    face_trace = box_trace(np.zeros(3), np.ones(3), draw_faces=True, color=2.0)
    traces = box_traces(
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
        color=np.array([0.1, 0.9]),
        name="box",
    )
    scatter = scatter_boxes(
        np.array([[0, 0, 0]], dtype=np.float32),
        dimension=np.array([1.0, 2.0, 3.0]),
        shared_legend=False,
    )
    boxes_with_hover = box_traces(
        np.array([[0, 0, 0]], dtype=np.float32),
        np.array([[1, 1, 1]], dtype=np.float32),
        hovertext=["box"],
    )

    assert edge_trace.type == "scatter3d"
    assert edge_with_text.hovertext == ("edge",) * 8
    assert face_trace.type == "mesh3d"
    assert face_trace.intensity.tolist() == [2.0] * 8
    assert len(traces) == 2
    assert len(scatter) == 1
    assert "box" in boxes_with_hover[0].hovertemplate
    assert np.isclose(max(scatter[0].z), 3.0)


def test_mesh_envelope_helpers_cover_numeric_and_string_colors():
    """Hull, ellipsoid, and cylinder helpers should build finite mesh traces."""
    hull = hull_trace(POINTS, color=3.0)
    hull_text = hull_trace(POINTS, hovertext=["a"] * len(POINTS))
    ell_from_points = ellipsoid_trace(points=POINTS, color="green")
    ell_from_one_point = ellipsoid_trace(points=POINTS[:1])
    ell_from_cov = ellipsoid_trace(centroid=np.zeros(3), covmat=np.eye(3))
    ell_with_text = ellipsoid_trace(
        centroid=np.zeros(3),
        covmat=np.eye(3),
        hovertext=["a"] * 100,
    )
    ell_list = ellipsoid_traces(
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        np.eye(3),
        color=np.array([1.0, 2.0]),
        hovertext=np.array(["a", "b"]),
        shared_legend=False,
    )
    cylinders = cylinder_traces(
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        axis=np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32),
        height=np.array([1.0, 2.0]),
        diameter=np.array([1.0, 2.0]),
        color=np.array([0.0, 1.0]),
        hovertext=np.array(["a", "b"]),
        shared_legend=False,
    )
    cylinders_auto_hover = cylinder_traces(
        np.array([[0, 0, 0]], dtype=np.float32),
        axis=np.array([0, 0, 1], dtype=np.float32),
        height=1.0,
        diameter=1.0,
        color=np.array([1.0]),
    )
    ell_auto_hover = ellipsoid_traces(
        np.array([[0, 0, 0]], dtype=np.float32),
        np.eye(3),
        color=np.array([1.0]),
    )
    cylinder_text = cylinder_traces(
        np.array([[0, 0, 0]], dtype=np.float32),
        axis=np.array([0, 0, 1], dtype=np.float32),
        height=1.0,
        diameter=1.0,
        hovertext=["cylinder"],
    )
    cylinder_direct_text = cylinder_trace(
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        1.0,
        1.0,
        hovertext=["c"] * 100,
    )
    cone_text = cone_trace(POINTS, hovertext=["cone"] * 100)

    assert hull.intensity.tolist() == [3.0] * len(POINTS)
    assert hull_text.hovertext == ("a",) * len(POINTS)
    assert ell_from_points.color == "green"
    assert np.all(np.isfinite(ell_from_one_point.x))
    assert np.all(np.isfinite(ell_from_cov.x))
    assert ell_with_text.hovertext == ("a",) * 100
    assert len(ell_list) == 2
    assert len(cylinders) == 2
    assert "Value:" in cylinders_auto_hover[0].hovertemplate
    assert "Value:" in ell_auto_hover[0].hovertemplate
    assert "cylinder" in cylinder_text[0].hovertemplate
    assert cylinder_direct_text.hovertext == ("c",) * 100
    assert cone_text.hovertext == ("cone",) * 100
    with pytest.raises(ValueError, match="either `points`"):
        ellipsoid_trace()
    with pytest.raises(ValueError, match="both `color` and `intensity`"):
        hull_trace(POINTS, color=1.0, intensity=np.ones(len(POINTS)))


def test_scatter_clusters_modes_and_hover_validation():
    """Cluster drawing should cover merged and split traces with validation."""
    merged = scatter_clusters(POINTS, CLUSTS, single_trace=True)
    circles = scatter_clusters(POINTS, CLUSTS, mode="circle", color=[0, 1])
    hulls = scatter_clusters(POINTS, CLUSTS, mode="hull", color=[0.5, 1.5])

    assert len(merged) == 1
    assert len(circles) == 2
    assert len(hulls) == 2
    with pytest.raises(ValueError, match="hovertext"):
        scatter_clusters(POINTS, CLUSTS, hovertext=["a", "b", "c", "d"])


def test_scatter_clusters_color_and_mode_branches():
    """Cluster drawing should handle scalar, per-point, and named split traces."""
    per_point = scatter_clusters(POINTS, CLUSTS, color=np.arange(len(POINTS)))
    per_point_hover = scatter_clusters(
        POINTS,
        CLUSTS,
        hovertext=np.array([f"p{i}" for i in range(len(POINTS))]),
        single_trace=True,
    )
    scalar_hover = scatter_clusters(POINTS, CLUSTS, hovertext="cluster")
    scalar = scatter_clusters(POINTS, CLUSTS, color="red", shared_legend=False)
    empty = scatter_clusters(
        POINTS,
        [np.empty(0, dtype=np.int64)],
        single_trace=True,
    )
    circle = scatter_clusters(POINTS, CLUSTS, mode="circle", single_trace=True)
    ellipsoid = scatter_clusters(
        POINTS,
        CLUSTS,
        mode="ellipsoid",
        color=[0.0, 1.0],
        name=["a", "b"],
        shared_legend=False,
    )
    cone = scatter_clusters(
        POINTS,
        CLUSTS,
        mode="cone",
        color=[0.0, 1.0],
        shared_legend=False,
    )

    assert len(per_point) == 2
    assert len(per_point_hover[0].x) == len(POINTS)
    assert "cluster" in scalar_hover[0].hovertemplate
    assert len(empty[0].x) == 0
    assert per_point[0].marker.color.tolist() == [0, 1, 2]
    assert scalar[1].marker.color == "red"
    assert len(circle) == 1
    assert ellipsoid[0].name == "a"
    assert len(cone) == 2
    with pytest.raises(ValueError, match="color"):
        scatter_clusters(POINTS, CLUSTS, color=[1, 2, 3])
    with pytest.raises(ValueError, match="not recognized"):
        scatter_clusters(POINTS, CLUSTS, mode="bad")


def test_network_helpers_build_topology_and_schematic():
    """Network helpers should draw nodes and edges in 3D and schematic views."""
    edges = np.array([[0, 1]], dtype=np.int64)

    topology = network_topology(POINTS, CLUSTS, edges, mode="scatter")
    topology_extra_cols = network_topology(
        np.hstack([np.zeros((len(POINTS), 1)), POINTS]),
        CLUSTS,
        edges,
        mode="scatter",
    )
    topology_with_labels = network_topology(
        POINTS,
        CLUSTS,
        edges,
        mode="scatter",
        edge_labels=np.array([1]),
    )
    schematic = network_schematic(CLUSTS, edges, np.array([0, 1]), edge_labels=[1])

    assert len(topology) == 2
    assert len(topology_extra_cols) == 2
    assert topology_with_labels[-1].line.color.tolist() == [1, 1, 1]
    assert len(topology[-1].x) == 3
    assert len(schematic) == 2
    assert schematic[0].type == "scatter"
    assert "Edge label" in schematic[1].text[0]
    with pytest.raises(ValueError, match="0 or 1"):
        network_schematic(CLUSTS, edges, np.array([0, 2]))


def test_network_topology_covers_centroid_and_cone_edges():
    """Network topology should draw centroid and cone-start edge variants."""
    edges = np.array([[0, 1]], dtype=np.int64)

    circles = network_topology(POINTS, CLUSTS, edges, mode="circle")
    hulls = network_topology(POINTS, CLUSTS, edges, mode="hull")
    cones = network_topology(POINTS, CLUSTS, edges, mode="cone")

    assert len(circles[-1].x) == 3
    assert len(hulls[-1].x) == 3
    assert len(cones[-1].x) == 3
    with pytest.raises(ValueError, match="clust_labels"):
        network_topology(POINTS, CLUSTS, edges, color="red")


def test_layout_and_evaluation_helpers():
    """Layout and evaluation helpers should return usable plot objects."""
    layout = layout3d(ranges=np.array([[0, 1], [0, 2], [0, 3]], dtype=float))
    fig = dual_figure3d([scatter_points(POINTS)[0]], [scatter_points(POINTS)[0]])
    width, height = set_latex_size(250)

    assert layout.scene.xaxis.range == (0, 1)
    assert len(fig.data) == 2
    assert width > height > 0
    assert color_rgba((1, 2, 3), 0.5) == "rgba(1, 2, 3, 0.5)"

    _, ax = plt.subplots()
    image = heatmap(np.array([[0.1, 0.9]]), ["r"], ["a", "b"], ax=ax)
    image_default_ax = heatmap(np.array([[0.1]]), ["r"], ["a"])
    texts = annotate_heatmap(image)
    texts_threshold = annotate_heatmap(image, threshold=0.5)
    texts_unc = annotate_heatmap(
        image,
        unc=np.array([[0.01, 0.02]]),
        valfmt="{x:.1f} +/- {unc:.2f}",
    )

    assert len(texts) == 2
    assert len(texts_threshold) == 2
    assert image_default_ax.axes is not None
    assert texts_unc[0].get_text() == "0.1 +/- 0.01"
    plt.close("all")


def test_layout3d_uses_point_ranges_meta_and_dark_options(monkeypatch):
    """Layout helper should infer ranges from points and metadata."""

    class MetaStub:
        lower = np.array([-1.0, -2.0, -3.0])
        upper = np.array([9.0, 8.0, 7.0])
        size = np.array([1.0, 2.0, 5.0])

        def to_px(self, points):
            return points

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
    with pytest.raises(ValueError, match="geo"):
        layout3d(ranges=np.array([[0, 1], [0, 1], [0, 1]]), geo=object())
    with pytest.raises(ValueError, match="metadata"):
        layout3d(
            geo=SimpleNamespace(get_boundaries=lambda **_: np.array([[0.0, 1.0]] * 3))
        )
    with pytest.raises(ValueError, match="ranges"):
        layout3d(ranges=np.array([[0, 1], [0, 1], [0, 1]]), meta=MetaStub())


def test_layout_helpers_cover_style_and_existing_layout(monkeypatch):
    """Layout helpers should cover style and existing-layout branches."""
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


def test_lite_helpers_draw_tracks_showers_interactions_and_legends():
    """Lite helpers should dispatch particles and interactions."""
    track = SimpleNamespace(
        shape=TRACK_SHP,
        start_point=np.zeros(3),
        end_point=np.ones(3),
        start_dir=np.array([0.0, 0.0, 1.0]),
        ke=100.0,
    )
    shower = SimpleNamespace(
        shape=0,
        start_point=np.zeros(3),
        end_point=np.ones(3),
        start_dir=np.array([0.0, 0.0, 1.0]),
        ke=100.0,
    )
    interaction = SimpleNamespace(particles=[track, shower])

    line = track_line_trace(track.start_point, track.end_point, color=1.0)
    line_text = track_line_trace(
        track.start_point, track.end_point, hovertext=["a", "b"]
    )
    line_custom = track_line_trace(
        track.start_point,
        track.end_point,
        line={"dash": "dash"},
        colorscale="Viridis",
        cmin=0.0,
        cmax=1.0,
    )
    particle_traces = scatter_lite_particles([track, shower], color=[0.0, 1.0])
    named_particles = scatter_lite_particles(
        [track, shower],
        color=[0.0, 1.0],
        hovertext=["t", "s"],
        name=["track", "shower"],
        shared_legend=False,
    )
    scalar_named_particles = scatter_lite_particles(
        [track, shower],
        name="particle",
        shared_legend=False,
    )
    interaction_traces = scatter_lite_interactions([interaction], color=[2.0])
    named_interactions = scatter_lite_interactions(
        [interaction],
        color=[2.0],
        hovertext=["i"],
        name=["interaction"],
        shared_legend=False,
    )
    scalar_named_interactions = scatter_lite_interactions(
        [interaction],
        name="interaction",
        shared_legend=False,
    )

    assert scatter_lite([]) == []
    assert len(scatter_lite([track])) == 2
    assert line.type == "scatter3d"
    assert line_text.hovertext == ("a", "b")
    assert line_custom.line.dash == "dash"
    assert legend_trace("red").marker.color == ("red",)
    assert len(particle_traces) == 3
    assert named_particles[0].name == "track"
    assert scalar_named_particles[0].name == "particle 0"
    assert len(interaction_traces) == 3
    assert named_interactions[0].name == "interaction"
    assert scalar_named_interactions[0].name == "interaction 0"
    assert len(scatter_lite([interaction])) == 3
    with pytest.raises(ValueError, match="single color"):
        track_line_trace(track.start_point, track.end_point, color=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="either `color` or `intensity`"):
        from spine.vis.lite import em_cone_trace

        em_cone_trace(
            shower.start_point,
            shower.start_dir,
            shower.ke,
            color=1.0,
            intensity=np.ones(4),
        )
    with pytest.raises(ValueError, match="single color"):
        from spine.vis.lite import em_cone_trace

        em_cone_trace(
            shower.start_point, shower.start_dir, shower.ke, color=np.arange(2)
        )
    from spine.vis.lite import em_cone_trace

    lite_cone_text = em_cone_trace(
        shower.start_point,
        shower.start_dir,
        shower.ke,
        hovertext=["lite"] * 100,
    )
    assert lite_cone_text.hovertext == ("lite",) * 100


def test_scatter_particles_uses_truth_particle_metadata():
    """Truth-particle drawing should build one trace per particle with voxels."""
    particle = Particle(id=0, group_id=1, interaction_id=2, pid=3, shape=4)
    labels = np.zeros((2, PART_COL + 1), dtype=np.float32)
    labels[:, :3] = POINTS[:2]
    labels[:, PART_COL] = 0

    traces = scatter_particles(labels, [particle])

    assert len(traces) == 1
    assert traces[0].name == "Particle 0"
    assert "Particle ID" in traces[0].hovertemplate


def test_train_drawer_finds_keys_and_loads_logs(tmp_path):
    """TrainDrawer should resolve aliases and combine train/validation logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    pd.DataFrame({"iter": [0, 1], "loss": [3.0, 2.0]}).to_csv(
        log_dir / "train-0.csv", index=False
    )
    pd.DataFrame({"iter": [2], "loss": [1.0]}).to_csv(
        log_dir / "train-2.csv", index=False
    )
    pd.DataFrame({"acc": [0.5, 0.7]}).to_csv(log_dir / "inference-2.csv", index=False)

    drawer = TrainDrawer(str(tmp_path), interactive=True)
    key, key_name = drawer.find_key({"fallback": []}, "loss:fallback")
    train_df = drawer.get_training_df(str(log_dir), ["loss"])
    val_df = drawer.get_validation_df(str(log_dir), ["accuracy:acc"])
    drawer.set_log_dir(str(log_dir))

    assert (key, key_name) == ("fallback", "loss")
    assert train_df["loss"].tolist() == [3.0, 2.0, 1.0]
    assert val_df["accuracy_mean"].tolist() == [0.6]
    assert drawer.log_dir == str(log_dir)
    with pytest.raises(KeyError, match="Could not find"):
        drawer.find_key({"loss": []}, "missing")
    with pytest.raises(FileNotFoundError, match="Found no train log"):
        drawer.get_training_df(str(tmp_path / "missing"), ["loss"])


def test_train_drawer_draws_interactive_and_matplotlib_logs(tmp_path, monkeypatch):
    """TrainDrawer.draw should exercise plotly and matplotlib rendering paths."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    pd.DataFrame(
        {"iter": [0, 1], "epoch": [0.0, 0.5], "loss": [2.0, 1.0], "acc": [0.4, 0.6]}
    ).to_csv(model_dir / "train-0.csv", index=False)
    pd.DataFrame({"loss": [1.5, 1.7], "acc": [0.5, 0.7]}).to_csv(
        model_dir / "inference-2.csv", index=False
    )
    monkeypatch.setattr("spine.vis.train.iplot", lambda fig: None)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    interactive = TrainDrawer(str(tmp_path), interactive=True)
    interactive.draw("model", ["loss", "acc"], same_plot=False, iter_per_epoch=2)

    matplotlib_drawer = TrainDrawer(str(tmp_path), interactive=False)
    matplotlib_drawer.draw(
        "model",
        "loss",
        smoothing=1,
        print_min=True,
        print_max=True,
        figure_name=str(tmp_path / "training"),
    )

    assert (tmp_path / "training.png").exists()


def test_train_drawer_draws_multi_model_matplotlib_branches(tmp_path, monkeypatch):
    """TrainDrawer should cover multi-model/multi-metric matplotlib branches."""
    for model, offset in [("model_a", 0.0), ("model_b", 1.0)]:
        model_dir = tmp_path / model
        model_dir.mkdir()
        pd.DataFrame(
            {
                "iter": [0, 1, 2, 3],
                "epoch": [0.0, 0.5, 1.0, 1.5],
                "loss": [3.0 + offset, 2.0 + offset, 1.5 + offset, 1.0 + offset],
                "acc": [0.1 + offset, 0.2 + offset, 0.3 + offset, 0.4 + offset],
            }
        ).to_csv(model_dir / "train-0.csv", index=False)
        pd.DataFrame({"loss": [1.5, 1.7], "acc": [0.5, 0.7]}).to_csv(
            model_dir / "inference-2.csv", index=False
        )

    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    drawer = TrainDrawer(str(tmp_path), interactive=False)
    drawer.draw(
        ["model_a", "model_b"],
        ["loss", "acc"],
        limits={"loss": [0.0, 4.0], "acc": [0.0, 2.0]},
        model_name={"model_a": "A", "model_b": "B"},
        metric_name={"loss": "Loss", "acc": "Accuracy"},
        same_plot=False,
        max_iter=3,
        step=2,
        smoothing=2,
    )


def test_train_drawer_validates_display_names(tmp_path):
    """TrainDrawer should reject ambiguous display-name inputs."""
    drawer = TrainDrawer(str(tmp_path), interactive=True)

    with pytest.raises(ValueError, match="model_name"):
        drawer.draw(["a", "b"], "loss", model_name="model")
    with pytest.raises(ValueError, match="metric_name"):
        drawer.draw("a", ["loss", "acc"], metric_name="metric")


def test_train_drawer_covers_name_limits_and_empty_log_branches(tmp_path, monkeypatch):
    """TrainDrawer should cover scalar names, list limits, and empty chunks."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    pd.DataFrame({"iter": [0, 1], "epoch": [0.0, 1.0], "loss": [2.0, 1.0]}).to_csv(
        model_dir / "train-0.csv", index=False
    )
    pd.DataFrame(columns=["iter", "epoch", "loss"]).to_csv(
        model_dir / "train-1.csv", index=False
    )
    monkeypatch.setattr("spine.vis.train.iplot", lambda fig: None)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    interactive = TrainDrawer(str(tmp_path), interactive=True)
    interactive.draw(
        "model",
        "loss",
        limits=[0.0, 3.0],
        model_name="Model",
        metric_name="Loss",
        same_plot=True,
    )

    matplotlib_drawer = TrainDrawer(str(tmp_path), interactive=False)
    matplotlib_drawer.draw(
        "model",
        "loss",
        limits=[0.0, 3.0],
        model_name="Model",
        metric_name="Loss",
        same_plot=True,
    )
    with matplotlib.rc_context():
        paper_drawer = TrainDrawer(str(tmp_path), interactive=False, paper=True)
        assert paper_drawer.linewidth == 0.5
