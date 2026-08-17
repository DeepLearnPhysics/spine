"""Tests for arrow visualization helpers."""

import numpy as np
import pytest

from spine.vis.trace.arrow import scatter_arrows

POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def test_scatter_arrows_builds_trunks_and_tips():
    traces = scatter_arrows(
        POINTS,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        color=["red", "blue"],
        hovertext=["a", "b"],
    )

    assert len(traces) == 3
    assert traces[0].mode == "lines"
    assert traces[1].type == "cone"
    assert traces[2].type == "cone"
    assert traces[1].colorscale[0][1] == "red"
    assert traces[2].colorscale[0][1] == "blue"


def test_scatter_arrows_handles_scalar_hovertext():
    traces = scatter_arrows(
        POINTS[:1],
        np.array([[1.0, 0.0, 0.0]]),
        hovertext="direction",
    )

    assert "direction" in traces[0].text[0]


def test_scatter_arrows_accepts_numpy_colorscale():
    """Per-arrow colors should accept array-valued discrete color scales."""
    traces = scatter_arrows(
        POINTS,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        color=np.array([0, 1]),
        colorscale=np.array(["#ff0000", "#0000ff"]),
        cmin=0,
        cmax=1,
    )

    assert traces[1].colorscale[0][1] == "rgb(255, 0, 0)"
    assert traces[2].colorscale[0][1] == "rgb(0, 0, 255)"


def test_scatter_arrows_batches_shared_tip_colors():
    """A shared arrow color should require only one cone trace."""
    count = 500
    traces = scatter_arrows(
        np.zeros((count, 3), dtype=np.float32),
        np.ones((count, 3), dtype=np.float32),
        color="red",
    )

    assert len(traces) == 2
    assert len(traces[1].x) == count


def test_scatter_arrows_groups_repeated_tip_colors():
    """Repeated per-arrow colors should share batched cone traces."""
    traces = scatter_arrows(
        np.zeros((4, 3), dtype=np.float32),
        np.ones((4, 3), dtype=np.float32),
        color=["red", "blue", "red", "blue"],
    )

    assert len(traces) == 3
    assert [len(trace.x) for trace in traces[1:]] == [2, 2]


def test_arrow_validation_rejects_mismatched_colors():
    with pytest.raises(ValueError, match="length must match"):
        scatter_arrows(
            POINTS,
            np.ones((2, 3), dtype=np.float32),
            color=np.array([1.0, 2.0, 3.0]),
        )
