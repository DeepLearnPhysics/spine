"""Tests for cylinder visualization helpers."""

import numpy as np
import pytest

from spine.vis.cylinder import cylinder_trace, cylinder_traces


def test_cylinder_trace_handles_antiparallel_axis():
    trace = cylinder_trace(
        centroid=np.zeros(3),
        axis=np.array([0.0, 0.0, -1.0]),
        height=4.0,
        diameter=2.0,
    )

    assert np.all(np.isfinite(trace.x))
    assert np.all(np.isfinite(trace.y))
    assert np.all(np.isfinite(trace.z))
    assert np.isclose(np.min(trace.z), -2.0)
    assert np.isclose(np.max(trace.z), 2.0)


def test_cylinder_traces_cover_vectorized_and_hover_paths():
    cylinders = cylinder_traces(
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        axis=np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32),
        height=np.array([1.0, 2.0]),
        diameter=np.array([1.0, 2.0]),
        color=np.array([0.0, 1.0]),
        hovertext=np.array(["a", "b"]),
        shared_legend=False,
    )
    auto_hover = cylinder_traces(
        np.array([[0, 0, 0]], dtype=np.float32),
        axis=np.array([0, 0, 1], dtype=np.float32),
        height=1.0,
        diameter=1.0,
        color=np.array([1.0]),
    )
    with_text = cylinder_traces(
        np.array([[0, 0, 0]], dtype=np.float32),
        axis=np.array([0, 0, 1], dtype=np.float32),
        height=1.0,
        diameter=1.0,
        hovertext=["cylinder"],
    )
    direct_text = cylinder_trace(
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        1.0,
        1.0,
        hovertext=["c"] * 100,
    )

    assert len(cylinders) == 2
    assert "Value:" in auto_hover[0].hovertemplate
    assert "cylinder" in with_text[0].hovertemplate
    assert direct_text.hovertext == ("c",) * 100


def test_cylinder_validation_paths():
    centroids = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

    with pytest.raises(ValueError, match="both `color` and `intensity`"):
        cylinder_trace(
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            1.0,
            1.0,
            color=1.0,
            intensity=np.ones(4),
        )
    with pytest.raises(ValueError, match="one color"):
        cylinder_traces(
            centroids, np.array([0.0, 0.0, 1.0]), 1.0, 1.0, color=np.arange(3)
        )
    with pytest.raises(ValueError, match="one hovertext"):
        cylinder_traces(
            centroids,
            np.array([0.0, 0.0, 1.0]),
            1.0,
            1.0,
            hovertext=np.arange(3),
        )
    with pytest.raises(ValueError, match="one axis"):
        cylinder_traces(centroids, np.ones((3, 3)), 1.0, 1.0)
    with pytest.raises(ValueError, match="one height"):
        cylinder_traces(centroids, np.array([0.0, 0.0, 1.0]), np.arange(3), 1.0)
    with pytest.raises(ValueError, match="one diameter"):
        cylinder_traces(centroids, np.array([0.0, 0.0, 1.0]), 1.0, np.arange(3))
