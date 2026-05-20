"""Tests for output object visualization."""

import numpy as np
import pytest

from spine.constants import TRACK_SHP
from spine.data.out import RecoInteraction, RecoParticle, TruthParticle
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
    with pytest.raises(AssertionError, match="Object type"):
        drawer.get("bad")
    with pytest.raises(ValueError, match="not available"):
        drawer.get("interactions", attr=["not_an_attr"])
    with pytest.raises(AssertionError, match="Interactions do not have"):
        drawer.get("interactions", draw_end_points=True)


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
