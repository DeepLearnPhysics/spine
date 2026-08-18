"""Tests for interaction-vertex reconstruction helpers."""

import numpy as np
import pytest

import spine.physics.vertex as vertex_module
from spine.constants import SHOWR_SHP, TRACK_SHP
from spine.physics.vertex import (
    angular_loss,
    get_confluence_points,
    get_pseudovertex,
    get_vertex,
    get_weighted_pseudovertex,
)


def arrays(points):
    """Convert coordinate literals to the float32 representation used by numba."""
    return np.asarray(points, dtype=np.float32)


def test_vertex_trivial_confluence_and_fallback_modes():
    """Vertex selection should report each common reconstruction fallback."""
    empty = arrays([]).reshape(0, 3)
    vertex, mode = get_vertex(empty, empty, empty, np.empty(0), return_mode=True)
    assert mode == "no_particle"
    assert np.isneginf(vertex).all()

    starts = arrays([[1, 2, 3]])
    vertex, mode = get_vertex(
        starts, starts, arrays([[1, 0, 0]]), [SHOWR_SHP], return_mode=True
    )
    np.testing.assert_array_equal(vertex, starts[0])
    assert mode == "single_start"

    starts = arrays([[0, 0, 0], [0.5, 0, 0]])
    ends = arrays([[5, 0, 0], [0.5, 5, 0]])
    directions = arrays([[1, 0, 0], [0, 1, 0]])
    vertex, mode = get_vertex(
        starts, ends, directions, np.array([TRACK_SHP, TRACK_SHP]), return_mode=True
    )
    assert mode == "confluence_nodir"

    starts = arrays([[0, 0, 0], [20, 10, 0]])
    ends = arrays([[10, 0, 0], [20, 20, 0]])
    vertex, mode = get_vertex(
        starts,
        ends,
        directions,
        np.array([TRACK_SHP, SHOWR_SHP]),
        return_mode=True,
    )
    assert mode == "confluence_track_showers"

    vertex, mode = get_vertex(
        starts, ends, directions, np.array([TRACK_SHP, TRACK_SHP]), return_mode=True
    )
    assert mode == "track_length"

    vertex, mode = get_vertex(
        starts,
        ends,
        directions,
        np.array([SHOWR_SHP, SHOWR_SHP]),
        anchor_vertex=False,
        return_mode=True,
    )
    assert mode == "pseudo_vertex"
    assert vertex.shape == (3,)


def test_vertex_math_helpers():
    """Angular losses, confluences, and weighted line fits should cover branches."""
    candidates = arrays([[0, 0, 0], [1, 0, 0]])
    points = arrays([[2, 1, 0]])
    directions = arrays([[1, 0, 0]])
    assert angular_loss(candidates, points, directions, True).shape == (2,)
    assert angular_loss(candidates, points, directions, False).shape == (2,)

    starts = arrays([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]])
    ends = arrays([[5, 0, 0], [0.5, 5, 0], [0, 0.5, 5]])
    assert len(get_confluence_points(starts, touching_threshold=1.0)) == 1
    assert len(get_confluence_points(starts, ends, touching_threshold=1.0)) == 1
    assert len(get_confluence_points(starts, touching_threshold=0.1)) == 0

    with pytest.raises(AssertionError, match="without points"):
        get_pseudovertex(arrays([]).reshape(0, 3), arrays([]).reshape(0, 3))
    np.testing.assert_array_equal(get_pseudovertex(starts[:1], directions), starts[0])

    # Parallel lines are singular and therefore use the barycenter fallback.
    parallel = arrays([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(
        get_pseudovertex(starts, parallel), np.mean(starts, axis=0)
    )
    weighted = get_weighted_pseudovertex(
        starts, parallel, np.array([1.0, 2.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(weighted, np.average(starts, axis=0, weights=[1, 2, 1]))

    nonparallel = arrays([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.isfinite(get_pseudovertex(starts, nonparallel)).all()
    assert np.isfinite(
        get_weighted_pseudovertex(starts, nonparallel, np.ones(3, dtype=np.float32))
    ).all()
    np.testing.assert_array_equal(
        get_weighted_pseudovertex(starts[:1], directions, np.ones(1, dtype=np.float32)),
        starts[0],
    )


def test_vertex_unannotated_returns_and_direction_confluence(monkeypatch):
    """Unannotated return paths and direction-disambiguated confluence are covered."""
    empty = arrays([]).reshape(0, 3)
    assert np.isneginf(get_vertex(empty, empty, empty, np.empty(0))).all()
    starts = arrays([[0, 0, 0], [10, 0, 0]])
    ends = arrays([[2, 0, 0], [12, 0, 0]])
    directions = arrays([[1, 0, 0], [1, 0, 0]])
    assert np.array_equal(
        get_vertex(starts[:1], ends[:1], directions[:1], np.array([TRACK_SHP])),
        starts[0],
    )

    calls = iter([[starts[0], starts[1]], [starts[0]]])
    monkeypatch.setattr(
        vertex_module, "get_confluence_points", lambda *args, **kwargs: next(calls)
    )
    vertex, mode = get_vertex(
        starts,
        ends,
        directions,
        np.array([TRACK_SHP, TRACK_SHP]),
        return_mode=True,
    )
    assert mode == "confluence_dir"
    np.testing.assert_array_equal(vertex, starts[0])

    calls = iter([[starts[0], starts[1]], [starts[0]]])
    monkeypatch.setattr(
        vertex_module, "get_confluence_points", lambda *args, **kwargs: next(calls)
    )
    np.testing.assert_array_equal(
        get_vertex(starts, ends, directions, np.array([TRACK_SHP, TRACK_SHP])),
        starts[0],
    )

    # Exercise the remaining unannotated selection modes.
    monkeypatch.undo()
    close_starts = arrays([[0, 0, 0], [0.5, 0, 0]])
    far_ends = arrays([[5, 0, 0], [0.5, 5, 0]])
    np.testing.assert_array_equal(
        get_vertex(
            close_starts,
            far_ends,
            directions,
            np.array([TRACK_SHP, TRACK_SHP]),
        ),
        np.mean(close_starts, axis=0),
    )
    mixed_starts = arrays([[0, 0, 0], [20, 10, 0]])
    mixed_ends = arrays([[10, 0, 0], [20, 20, 0]])
    get_vertex(mixed_starts, mixed_ends, directions, np.array([TRACK_SHP, SHOWR_SHP]))
    get_vertex(mixed_starts, mixed_ends, directions, np.array([TRACK_SHP, TRACK_SHP]))
    get_vertex(
        mixed_starts,
        mixed_ends,
        directions,
        np.array([SHOWR_SHP, SHOWR_SHP]),
        anchor_vertex=False,
    )
