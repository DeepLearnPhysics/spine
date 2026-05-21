"""Tests for box visualization helpers."""

import numpy as np
import pytest

from spine.vis.box import box_trace, box_traces, scatter_boxes


def test_box_helpers_draw_edges_faces_and_scatter_boxes():
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


def test_box_validation_rejects_bad_bounds_and_color_conflicts():
    with pytest.raises(ValueError, match="3 values"):
        box_trace(np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="greater"):
        box_trace(np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="Must not specify `line`"):
        box_trace(np.zeros(3), np.ones(3), line={}, color="red")
    with pytest.raises(ValueError, match="both `color` and `intensity`"):
        box_trace(
            np.zeros(3),
            np.ones(3),
            draw_faces=True,
            color=1.0,
            intensity=np.ones(8),
        )
    with pytest.raises(ValueError, match="upper boundary"):
        box_traces(np.zeros((2, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="one color"):
        box_traces(np.zeros((1, 3)), np.ones((1, 3)), color=np.arange(2))
    with pytest.raises(ValueError, match="one hovertext"):
        box_traces(np.zeros((1, 3)), np.ones((1, 3)), hovertext=np.arange(2))
    with pytest.raises(ValueError, match="three dimensions"):
        scatter_boxes(np.zeros((1, 3)), dimension=np.ones(2))
