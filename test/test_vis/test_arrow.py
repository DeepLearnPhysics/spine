"""Tests for arrow visualization helpers."""

import numpy as np
import pytest

from spine.vis.arrow import scatter_arrows

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

    assert len(traces) == 2
    assert traces[0].mode == "lines"
    assert traces[1].type == "cone"
    assert len(traces[1].x) == 2


def test_scatter_arrows_handles_scalar_hovertext():
    traces = scatter_arrows(
        POINTS[:1],
        np.array([[1.0, 0.0, 0.0]]),
        hovertext="direction",
    )

    assert "direction" in traces[0].text[0]


def test_arrow_validation_rejects_mismatched_colors():
    with pytest.raises(ValueError, match="length must match"):
        scatter_arrows(
            POINTS,
            np.ones((2, 3), dtype=np.float32),
            color=np.array([1.0, 2.0, 3.0]),
        )
