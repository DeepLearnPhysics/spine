"""Tests for lite visualization helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import TRACK_SHP
from spine.vis.lite import (
    em_cone_trace,
    legend_trace,
    scatter_lite,
    scatter_lite_interactions,
    scatter_lite_particles,
    track_line_trace,
)


def _track():
    return SimpleNamespace(
        shape=TRACK_SHP,
        start_point=np.zeros(3),
        end_point=np.ones(3),
        start_dir=np.array([0.0, 0.0, 1.0]),
        ke=100.0,
    )


def _shower():
    return SimpleNamespace(
        shape=0,
        start_point=np.zeros(3),
        end_point=np.ones(3),
        start_dir=np.array([0.0, 0.0, 1.0]),
        ke=100.0,
    )


def test_lite_helpers_draw_tracks_showers_interactions_and_legends():
    track = _track()
    shower = _shower()
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
    lite_cone_text = em_cone_trace(
        shower.start_point,
        shower.start_dir,
        shower.ke,
        hovertext=["lite"] * 100,
    )
    colored_cone = em_cone_trace(
        shower.start_point,
        np.array([0.0, 0.0, -1.0]),
        shower.ke,
        color="blue",
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
    assert lite_cone_text.hovertext == ("lite",) * 100
    assert colored_cone.color == "blue"


def test_lite_validation_rejects_mismatched_colors():
    track = _track()
    shower = _shower()
    interaction = SimpleNamespace(particles=[track])

    with pytest.raises(ValueError, match="one per particle"):
        scatter_lite_particles([track], color=[1.0, 2.0])
    with pytest.raises(ValueError, match="one per interaction"):
        scatter_lite_interactions([interaction], color=[1.0, 2.0])
    with pytest.raises(ValueError, match="single color"):
        track_line_trace(track.start_point, track.end_point, color=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="either `color` or `intensity`"):
        em_cone_trace(
            shower.start_point,
            shower.start_dir,
            shower.ke,
            color=1.0,
            intensity=np.ones(4),
        )
    with pytest.raises(ValueError, match="single color"):
        em_cone_trace(
            shower.start_point, shower.start_dir, shower.ke, color=np.arange(2)
        )
