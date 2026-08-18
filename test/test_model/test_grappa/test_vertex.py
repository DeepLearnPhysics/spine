"""Tests for shared GrapPA vertex-position decoding."""

import numpy as np
import pytest
import torch

from spine.data import Meta, TensorBatch
from spine.model.grappa.vertex import (
    decode_vertex_positions,
    vertex_position_scales,
)


def make_meta(count: int) -> Meta:
    """Build cubic image metadata with a configurable side length."""
    return Meta(
        lower=np.zeros(3),
        upper=np.full(3, count, dtype=np.float64),
        size=np.ones(3),
        count=np.full(3, count, dtype=np.int64),
    )


def test_vertex_position_scales_follow_event_counts() -> None:
    """Normalization scales should repeat each event's image dimensions."""
    prediction = TensorBatch(torch.zeros((3, 3)), counts=[2, 1])

    scales = vertex_position_scales(prediction, [make_meta(10), make_meta(20)])

    assert scales.tolist() == [[10.0] * 3, [10.0] * 3, [20.0] * 3]
    with pytest.raises(ValueError, match="one metadata entry"):
        vertex_position_scales(prediction, [make_meta(10)])


def test_decode_vertex_positions_restores_normalized_coordinates() -> None:
    """Normalized predictions should optionally return to image coordinates."""
    prediction = TensorBatch(
        torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]]),
        counts=[1, 1],
    )

    decoded = decode_vertex_positions(
        prediction,
        meta=[make_meta(10), make_meta(20)],
        normalize_positions=True,
        restore_absolute=True,
    )

    np.testing.assert_allclose(
        decoded.torch_tensor().detach().cpu().numpy(),
        [[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]],
    )


def test_decode_vertex_positions_uses_closest_anchor() -> None:
    """Anchor mode should add offsets to the closest particle endpoint."""
    prediction = TensorBatch(torch.tensor([[0.1, 0.1, 0.1]]), counts=[1])
    starts = TensorBatch(torch.zeros((1, 3)), counts=[1])
    ends = TensorBatch(torch.full((1, 3), 10.0), counts=[1])

    decoded = decode_vertex_positions(
        prediction,
        start_points=starts,
        end_points=ends,
        use_anchor_points=True,
    )

    np.testing.assert_allclose(
        decoded.torch_tensor().detach().cpu().numpy(), [[0.1, 0.1, 0.1]]
    )


def test_decode_vertex_positions_normalizes_anchor_points() -> None:
    """Normalized anchor offsets should decode back to absolute coordinates."""
    prediction = TensorBatch(torch.full((1, 3), 0.1), counts=[1])
    starts = TensorBatch(torch.zeros((1, 3)), counts=[1])
    ends = TensorBatch(torch.full((1, 3), 10.0), counts=[1])

    decoded = decode_vertex_positions(
        prediction,
        start_points=starts,
        end_points=ends,
        meta=[make_meta(10)],
        normalize_positions=True,
        use_anchor_points=True,
        restore_absolute=True,
    )

    np.testing.assert_allclose(
        decoded.torch_tensor().detach().cpu().numpy(), [[1.0, 1.0, 1.0]]
    )


@pytest.mark.parametrize(
    ("prediction", "kwargs", "message"),
    [
        (torch.zeros((2, 2)), {}, r"shape \(P, 3\)"),
        (torch.zeros((2, 3)), {"normalize_positions": True}, "requires `meta`"),
        (
            torch.zeros((2, 3)),
            {"position_scales": torch.ones((1, 3))},
            "scales must match",
        ),
        (
            torch.zeros((2, 3)),
            {"use_anchor_points": True},
            "requires particle end points",
        ),
    ],
)
def test_decode_vertex_positions_validates_inputs(prediction, kwargs, message) -> None:
    """Malformed decoding inputs should fail with actionable errors."""
    batch = TensorBatch(prediction, counts=[len(prediction)])
    with pytest.raises(ValueError, match=message):
        decode_vertex_positions(batch, **kwargs)


def test_decode_vertex_positions_validates_anchor_alignment() -> None:
    """Endpoint rows must align with the particle regression rows."""
    prediction = TensorBatch(torch.zeros((2, 3)), counts=[2])
    endpoints = TensorBatch(torch.zeros((1, 3)), counts=[1])

    with pytest.raises(ValueError, match="endpoint rows"):
        decode_vertex_positions(
            prediction,
            start_points=endpoints,
            end_points=endpoints,
            use_anchor_points=True,
        )
