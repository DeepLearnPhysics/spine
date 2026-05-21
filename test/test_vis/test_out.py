"""Tests for output object visualization."""

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import TRACK_SHP
from spine.data.larcv import CRTHit, Flash
from spine.data.out import (
    RecoInteraction,
    RecoParticle,
    TruthInteraction,
    TruthParticle,
)
from spine.geo.base import Geometry
from spine.vis import Drawer


def test_drawer_accepts_derived_hover_attr():
    """Drawer validation should accept derived DataBase attributes."""
    data = {
        "points": np.empty((0, 3), dtype=np.float32),
        "reco_particles": [RecoParticle()],
    }

    figure = Drawer(data, draw_mode="reco").get("particles", attr=["ke"])

    assert figure is not None


def test_drawer_accepts_skipped_hover_attr():
    """Drawer validation should still accept skipped DataBase attributes."""
    data = {
        "points": np.empty((0, 3), dtype=np.float32),
        "reco_particles": [RecoParticle()],
    }

    figure = Drawer(data, draw_mode="reco").get("particles", attr=["depositions"])

    assert figure is not None


def test_drawer_draws_reco_particles_with_auxiliary_traces():
    """Drawer should combine object, raw, endpoint, and direction traces."""
    particle = RecoParticle(
        id=0,
        index=np.array([0, 1], dtype=np.int32),
        shape=TRACK_SHP,
        pid=2,
        start_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        end_point=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        start_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    data = {
        "points": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "depositions": np.array([1.0, 2.0], dtype=np.float32),
        "reco_particles": [particle],
    }

    figure = Drawer(data, draw_mode="reco").get(
        "particles",
        attr=["pid"],
        color_attr="pid",
        draw_raw=True,
        draw_end_points=True,
        draw_directions=True,
    )

    assert len(figure.data) >= 6
    assert figure.data[0].name == "Raw input"


def test_drawer_draws_truth_long_form_attributes():
    """Drawer should support truth point/deposition long-form attributes."""
    particle = TruthParticle(
        id=0,
        index=np.array([0, 1], dtype=np.int32),
        index_adapt=np.array([0, 1], dtype=np.int32),
        points_adapt=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        depositions_adapt=np.array([1.0, 2.0], dtype=np.float32),
        depositions_adapt_q=np.array([0.5, 1.5], dtype=np.float32),
    )
    data = {
        "points": particle.points_adapt,
        "points_label": particle.points_adapt,
        "depositions_label": particle.depositions_adapt,
        "depositions_label_adapt": particle.depositions_adapt,
        "depositions_q_label": particle.depositions_adapt_q,
        "truth_particles": [particle],
    }

    figure = Drawer(data, draw_mode="truth", truth_point_mode="points_adapt").get(
        "particles",
        attr=["depositions_adapt"],
        color_attr="depositions_adapt",
    )

    assert len(figure.data) == 1
    assert "Deposition" in figure.data[0].text[0]


def test_drawer_draws_vertices_and_validates_requests():
    """Drawer should draw interaction vertices and reject invalid requests."""
    interaction = RecoInteraction(
        id=0,
        index=np.array([0], dtype=np.int32),
        vertex=np.array([0.0, 1.0, 2.0], dtype=np.float32),
    )
    data = {
        "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "reco_interactions": [interaction],
    }
    drawer = Drawer(data, draw_mode="reco")

    figure = drawer.get("interactions", draw_vertices=True)

    assert len(figure.data) == 2
    with pytest.raises(ValueError, match="Object type"):
        drawer.get("bad")
    with pytest.raises(ValueError, match="not available"):
        drawer.get("interactions", attr=["not_an_attr"])
    with pytest.raises(ValueError, match="Interactions do not have"):
        drawer.get("interactions", draw_end_points=True)
    with pytest.raises(ValueError, match="Interactions do not have"):
        drawer.get("interactions", draw_directions=True)


def test_drawer_low_level_format_helpers():
    """Low-level Drawer helpers should format known long-form attributes."""
    drawer = Drawer(
        {
            "points": np.empty((0, 3), dtype=np.float32),
            "reco_particles": [],
        },
        draw_mode="reco",
    )

    assert drawer.point_modes["points"] == "points_label"
    assert drawer.dep_modes["depositions"] == "depositions_label"
    assert drawer.source_modes["sources"] == "sources_label"
    assert drawer._is_depositions("depositions")
    assert drawer._is_sources("sources")
    assert drawer._dep_tostr(1.234).startswith("<br>Deposition")
    assert drawer._src_tostr(np.array([2, 3])) == "<br>Module, TPC: 2, 3"


def test_drawer_constructor_and_request_validation():
    """Drawer should validate draw modes, truth modes, and required products."""
    with pytest.raises(ValueError, match="mode"):
        Drawer({}, draw_mode="bad")
    with pytest.raises(ValueError, match="truth_point_mode"):
        Drawer({}, truth_point_mode="bad")
    with pytest.raises(ValueError, match="truth_dep_mode"):
        Drawer({}, truth_dep_mode="bad")

    drawer = Drawer(
        {"points": np.empty((0, 3)), "reco_particles": []}, draw_mode="reco"
    )
    with pytest.raises(ValueError, match="reco_particles"):
        Drawer({"points": np.empty((0, 3))}, draw_mode="reco").get("particles")
    with pytest.raises(ValueError, match="color scale"):
        drawer.get("particles", attr=["pid"], color_attr="shape")
    with pytest.raises(ValueError, match="titles"):
        drawer.get("particles", titles=["a"], split_traces=False)


def test_drawer_raw_and_lite_validation_paths():
    """Drawer should reject incompatible raw and lite draw requests."""
    particle = RecoParticle(
        id=0,
        index=np.array([0], dtype=np.int32),
        shape=TRACK_SHP,
        start_point=np.zeros(3),
        end_point=np.ones(3),
        start_dir=np.array([1.0, 0.0, 0.0]),
    )
    lite_drawer = Drawer({"reco_particles": [particle]}, draw_mode="reco", lite=True)

    with pytest.raises(RuntimeError, match="raw input"):
        lite_drawer.get("particles", draw_raw=True)
    with pytest.raises(ValueError, match="Long-form"):
        lite_drawer.get("particles", attr=["depositions"])
    with pytest.raises(ValueError, match="points"):
        Drawer({"reco_particles": [particle]}, draw_mode="reco").get("particles")


def test_drawer_color_helper_branches():
    """Drawer color helper should cover supported scalar attribute families."""
    particles = [
        RecoParticle(
            id=10 + i,
            index=np.array([i], dtype=np.int32),
            shape=i,
            pid=i,
            is_primary=bool(i % 2),
            depositions=np.array([float(i + 1)], dtype=np.float32),
        )
        for i in range(2)
    ]
    data = {
        "points": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "depositions": np.array([1.0, 2.0], dtype=np.float32),
        "reco_particles": particles,
    }
    drawer = Drawer(data, draw_mode="reco")

    pid_colors = drawer._object_colors("reco_particles", {"pid"}, "pid", False)
    primary_colors = drawer._object_colors(
        "reco_particles", {"is_primary"}, "is_primary", False
    )
    sum_colors = drawer._object_colors(
        "reco_particles", {"depositions_sum"}, "depositions_sum", False
    )
    id_colors = drawer._object_colors("reco_particles", {"id"}, "id", True)

    assert pid_colors["cmin"] == -1
    assert primary_colors["cmax"] == 1
    assert sum_colors["cmax"] == 2.0
    assert id_colors["name"] == "Reco particle"
    with pytest.raises(ValueError, match="not supported"):
        drawer._object_colors("reco_particles", {"ke"}, "ke", False)


def test_drawer_sources_and_split_auxiliary_traces():
    """Drawer should cover source colors and split auxiliary traces."""
    geo = Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
    )
    particle = RecoParticle(
        id=0,
        index=np.array([0], dtype=np.int32),
        sources=np.array([[0, 1]], dtype=np.int32),
        shape=TRACK_SHP,
        start_point=np.zeros(3),
        end_point=np.ones(3),
        start_dir=np.array([1.0, 0.0, 0.0]),
    )
    shower = RecoParticle(
        id=1,
        index=np.array([1], dtype=np.int32),
        sources=np.array([[0, 0]], dtype=np.int32),
        shape=0,
        start_point=np.ones(3),
        end_point=np.ones(3) * 2,
        start_dir=np.array([0.0, 1.0, 0.0]),
    )
    data = {
        "points": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        "depositions": np.array([1.0, 2.0], dtype=np.float32),
        "reco_particles": [particle, shower],
    }
    drawer = Drawer(data, draw_mode="reco", geo=geo)
    figure = drawer.get(
        "particles",
        attr=["sources"],
        color_attr="sources",
        draw_end_points=True,
        draw_directions=True,
        split_traces=True,
    )

    assert len(figure.data) > 4
    assert drawer.get_index(particle).tolist() == [0]


def test_drawer_raw_and_source_error_branches():
    """Drawer helper branches should reject missing raw/source context."""
    particle = RecoParticle(id=0, index=np.array([0], dtype=np.int32))
    drawer = Drawer(
        {
            "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "reco_particles": [particle],
        },
        draw_mode="reco",
    )

    with pytest.raises(ValueError, match="points"):
        drawer._raw_trace("reco")
    with pytest.raises(ValueError, match="Prefix"):
        drawer._raw_trace("bad")
    with pytest.raises(ValueError, match="color scale"):
        drawer._object_colors("reco_particles", set(), "sources", False)
    with pytest.raises(ValueError, match="sources"):
        drawer._object_colors("reco_particles", {"sources"}, "sources", False)

    truth_drawer = Drawer({"truth_particles": []}, draw_mode="truth")
    with pytest.raises(ValueError, match="points_label"):
        truth_drawer._raw_trace("truth")


def test_drawer_truth_interaction_enum_hover_value():
    """Truth interaction enum formatting should handle missing enum values."""
    from spine.constants.enums import LArSoftNuInteractionType, NuInteractionScheme

    interaction = TruthInteraction(
        id=0,
        interaction_type=int(LArSoftNuInteractionType.QE),
        interaction_scheme=int(NuInteractionScheme.LARSOFT),
    )
    drawer = Drawer({"truth_interactions": [interaction]}, draw_mode="truth")

    assert (
        drawer._enum_name(interaction, "interaction_type", interaction.interaction_type)
        == "QE"
    )
    assert drawer._format_hover_value(
        interaction, "interaction_type", interaction.interaction_type
    ).startswith("QE")
    interaction.interaction_scheme = -1
    assert (
        drawer._enum_name(interaction, "interaction_type", interaction.interaction_type)
        is None
    )


def test_drawer_source_colorscale_edge_counts():
    """Source colorscales should handle zero, one, and repeated color ranges."""
    particle = RecoParticle(
        id=0,
        index=np.array([0], dtype=np.int32),
        sources=np.array([[0, 0]], dtype=np.int32),
    )
    drawer = Drawer(
        {
            "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "reco_particles": [particle],
        },
        draw_mode="reco",
    )

    for count in (0, 1, 100):
        drawer.geo = type(
            "GeoStub",
            (),
            {
                "tpc": type("TPCStub", (), {"num_chambers": count})(),
                "get_sources": staticmethod(lambda sources: sources),
                "get_chambers": staticmethod(lambda sources: np.zeros(len(sources))),
            },
        )()
        colors = drawer._object_colors("reco_particles", {"sources"}, "sources", False)
        assert colors["cmax"] == count - 1


def test_drawer_id_colorscale_repeats_for_many_unique_ids():
    """ID colorscales should repeat when unique IDs exceed the base palette."""
    particles = [
        RecoParticle(id=i, index=np.array([0], dtype=np.int32)) for i in range(60)
    ]
    drawer = Drawer(
        {
            "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "reco_particles": particles,
        },
        draw_mode="reco",
    )

    colors = drawer._object_colors("reco_particles", {"id"}, "id", False)

    assert len(colors["colorscale"]) == 60


def test_drawer_split_scene_and_missing_auxiliary_interactions():
    """Drawer should cover split-scene geometry and missing auxiliary inputs."""
    geo = Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
    )
    data = {
        "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "points_label": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "reco_particles": [RecoParticle(id=0, index=np.array([0], dtype=np.int32))],
        "truth_particles": [
            TruthParticle(
                id=0,
                index=np.array([0], dtype=np.int32),
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            )
        ],
        "flashes": [],
    }
    drawer = Drawer(data, draw_mode="both", geo=geo, split_scene=True)
    figure = drawer.get("particles")

    assert len(figure.data) >= 2
    with pytest.raises(ValueError, match="vertices"):
        drawer.get("particles", draw_vertices=True)
    with pytest.raises(ValueError, match="matched flashes"):
        drawer.get("particles", draw_flashes=True)
    with pytest.raises(RuntimeError, match="CRT"):
        drawer._crt_trace("reco_particles", matched_only=False)


def test_drawer_adds_geo_traces_without_split_scene_and_rejects_titles():
    """Non-split scenes should still include detector traces on the last prefix."""
    geo = Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
    )
    data = {
        "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "points_label": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "reco_particles": [RecoParticle(id=0, index=np.array([0], dtype=np.int32))],
        "truth_particles": [
            TruthParticle(
                id=0,
                index=np.array([0], dtype=np.int32),
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            )
        ],
    }
    drawer = Drawer(data, draw_mode="both", geo=geo, split_scene=False)

    figure = drawer.get("particles")

    assert len(figure.data) >= 4
    with pytest.raises(ValueError, match="titles"):
        drawer.get("particles", titles=["Reco", "Truth"])


def test_drawer_truth_mode_compatibility_validation():
    """Truth point/deposition/source modes must be compatible."""
    particle = TruthParticle(
        id=0,
        index=np.array([0], dtype=np.int32),
        index_adapt=np.array([0], dtype=np.int32),
        points_adapt=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        depositions=np.array([1.0], dtype=np.float32),
        depositions_adapt=np.array([1.0], dtype=np.float32),
        sources=np.array([[0, 0]], dtype=np.int32),
        sources_adapt=np.array([[0, 0]], dtype=np.int32),
    )
    data = {
        "points": particle.points_adapt,
        "points_label": particle.points_adapt,
        "points_label_adapt": particle.points_adapt,
        "depositions_label": particle.depositions,
        "depositions_label_adapt": particle.depositions_adapt,
        "truth_particles": [particle],
    }
    geo = Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
    )
    drawer = Drawer(data, draw_mode="truth", truth_point_mode="points_adapt", geo=geo)

    with pytest.raises(ValueError, match="incompatible"):
        drawer.get("particles", attr=["depositions"], color_attr="depositions")
    with pytest.raises(ValueError, match="incompatible"):
        drawer.get("particles", attr=["sources"], color_attr="sources")
    drawer_no_geo = Drawer(data, draw_mode="truth", truth_point_mode="points_adapt")
    with pytest.raises(ValueError, match="geometry"):
        drawer_no_geo.get(
            "particles", attr=["sources_adapt"], color_attr="sources_adapt"
        )


def test_drawer_truth_raw_trace_and_empty_truth_auxiliaries():
    """Truth helpers should draw raw input and skip empty truth objects."""
    truth_particle = SimpleNamespace(
        is_truth=True,
        shape=TRACK_SHP,
        index=np.array([], dtype=np.int32),
        end_point=np.ones(3, dtype=np.float32),
        start_point=np.zeros(3, dtype=np.float32),
        start_dir=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    data = {
        "points_label": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "depositions_label": np.array([1.0], dtype=np.float32),
        "truth_particles": [truth_particle],
    }
    drawer = Drawer(data, draw_mode="truth")

    raw_trace = drawer._raw_trace("truth")
    end_points = drawer._point_trace("truth_particles", "end_point", split_traces=True)
    directions = drawer._direction_trace("truth_particles", split_traces=True)

    assert len(raw_trace) == 1
    assert end_points == []
    assert directions == []


def test_drawer_geo_and_auxiliary_validation_paths():
    """Drawer should validate flash/CRT requests that require geometry."""
    interaction = RecoInteraction(id=0, index=np.array([0], dtype=np.int32))
    data = {
        "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "reco_interactions": [interaction],
    }
    drawer = Drawer(data, draw_mode="reco")

    with pytest.raises(ValueError, match="flashes"):
        drawer.get("interactions", draw_flashes=True)
    with pytest.raises(ValueError, match="crthits"):
        drawer.get("interactions", draw_crthits=True)
    with pytest.raises(RuntimeError, match="optical"):
        drawer._flash_trace("reco_interactions", matched_only=False)
    with pytest.raises(RuntimeError, match="CRT"):
        drawer._crt_trace("reco_interactions", matched_only=False)

    geo = Geometry(
        name="demo",
        tag="v1",
        version=1,
        tpc={
            "dimensions": [10.0, 20.0, 30.0],
            "positions": [[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            "module_ids": [0, 0],
        },
    )
    geo_drawer = Drawer(data, draw_mode="reco", geo=geo)
    with pytest.raises(RuntimeError, match="optical"):
        geo_drawer._flash_trace("reco_interactions", matched_only=False)


def test_drawer_draws_matched_flashes_and_crt_hits():
    """Drawer should draw matched optical flashes and CRT hits with geometry."""
    geo = Geometry(
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
            "det_ids": [0, 0],
        },
        crt={
            "dimensions": [[2.0, 2.0, 2.0]],
            "positions": [[0.0, 30.0, 0.0]],
            "normals": [1],
        },
    )
    particle = RecoParticle(
        id=0,
        is_crt_matched=True,
        crt_ids=np.array([0], dtype=np.int32),
    )
    interaction = RecoInteraction(
        id=0,
        index=np.array([0], dtype=np.int32),
        is_flash_matched=True,
        flash_ids=np.array([0], dtype=np.int32),
        particles=[particle],
    )
    data = {
        "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "reco_interactions": [interaction],
        "flashes": [Flash(id=0, volume_id=0, pe_per_ch=np.array([1.0, 2.0]))],
        "crthits": [CRTHit(id=0, plane=0, center=np.array([0.0, 30.0, 0.0]))],
    }

    figure = Drawer(data, draw_mode="reco", geo=geo).get(
        "interactions",
        draw_flashes=True,
        draw_crthits=True,
    )

    assert any("flashes" in (trace.name or "") for trace in figure.data)
    assert any("CRT hits" in (trace.name or "") for trace in figure.data)


def test_drawer_draws_unmatched_flashes_and_crt_empty_plane_errors():
    """Flash and CRT helpers should cover unmatched flashes and missing CRT planes."""
    geo = Geometry(
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
            "det_ids": [0, 0],
        },
    )
    interaction = RecoInteraction(id=0, index=np.array([0], dtype=np.int32))
    flashes = [Flash(id=0, volume_id=0, pe_per_ch=np.array([1.0, 2.0]))]
    drawer = Drawer(
        {
            "points": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "reco_interactions": [interaction],
            "flashes": flashes,
            "crthits": [],
        },
        draw_mode="reco",
        geo=geo,
    )

    traces = drawer._flash_trace("reco_interactions", matched_only=False)

    assert traces
    with pytest.raises(RuntimeError, match="CRT planes"):
        drawer._crt_trace("reco_interactions", matched_only=False)
