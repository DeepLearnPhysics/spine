"""Tests for GrapPA graph augmentations."""

import numpy as np
import pytest
import torch

from spine.data import EdgeIndexBatch, TensorBatch
from spine.model.grappa.augment import EdgeDropout, EdgeSelection


def test_edge_dropout_validates_probability() -> None:
    """Reject probabilities outside the closed unit interval."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        EdgeDropout(-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        EdgeDropout(1.1)


def test_edge_dropout_keeps_reciprocal_pairs(monkeypatch) -> None:
    """Sample one decision per adjacent undirected edge pair and event."""
    edge_index = EdgeIndexBatch(
        np.array(
            [
                [0, 1, 0, 2, 3, 4],
                [1, 0, 2, 0, 4, 3],
            ]
        ),
        counts=[4, 2],
        spans=[3, 2],
        directed=False,
    )
    samples = iter((np.array([0.2, 0.8]), np.array([0.9])))
    monkeypatch.setattr(np.random, "random", lambda _: next(samples))

    selection = EdgeDropout(0.5)(edge_index)
    filtered = selection.filter_edge_index(edge_index)
    keep = selection.keep

    np.testing.assert_array_equal(keep.data, [False, False, True, True, True, True])
    np.testing.assert_array_equal(filtered.counts, [2, 2])
    np.testing.assert_array_equal(
        filtered.index,
        np.array([[0, 2, 3, 4], [2, 0, 4, 3]]),
    )


def test_edge_dropout_samples_directed_edges_on_original_backend(monkeypatch) -> None:
    """Directed edges are independent and Torch graphs stay Torch-backed."""
    edge_index = EdgeIndexBatch(
        torch.tensor([[0, 1, 2], [1, 2, 0]]),
        counts=[3, 0],
        spans=[3, 1],
        directed=True,
    )
    samples = iter((np.array([0.1, 0.6, 0.9]), np.empty(0)))
    monkeypatch.setattr(np.random, "random", lambda _: next(samples))

    selection = EdgeDropout(0.5)(edge_index)
    filtered = selection.filter_edge_index(edge_index)
    keep = selection.keep

    assert isinstance(filtered.index, torch.Tensor)
    assert filtered.counts.tolist() == [2, 0]
    assert filtered.index.tolist() == [[1, 2], [2, 0]]
    np.testing.assert_array_equal(keep.data, [False, True, True])


@pytest.mark.parametrize(
    "index, match",
    [
        (np.array([[0], [1]]), "even edge count"),
        (
            np.array([[0, 1], [1, 2]]),
            "adjacent reciprocal edge pairs",
        ),
    ],
)
def test_edge_dropout_rejects_malformed_undirected_graphs(index, match) -> None:
    """Materialized undirected inputs must honor GrapPA's pair convention."""
    edge_index = EdgeIndexBatch(
        index, counts=[index.shape[1]], spans=[3], directed=True
    )
    edge_index.directed = False

    with pytest.raises(ValueError, match=match):
        EdgeDropout(0.5)(edge_index)


def test_filter_tensor_batch_preserves_metadata_and_empty_events() -> None:
    """Selections recompute counts while retaining the logical tensor schema."""
    batch = TensorBatch(torch.arange(8).reshape(4, 2), counts=[2, 0, 2])
    keep = TensorBatch(
        torch.tensor([True, False, False, True]),
        counts=[2, 0, 2],
    )

    filtered = EdgeSelection(keep).filter_tensor(batch)

    assert filtered.counts.tolist() == [1, 0, 1]
    assert filtered.data.tolist() == [[0, 1], [6, 7]]
    assert filtered.schema == batch.schema

    with pytest.raises(ValueError, match="must align"):
        EdgeSelection(TensorBatch(np.ones(4), counts=[1, 3])).filter_tensor(batch)

    empty = TensorBatch(torch.empty((0, 2)), counts=[0, 0])
    empty_keep = TensorBatch(np.empty(0, dtype=bool), counts=[0, 0])
    assert EdgeSelection(empty_keep).filter_tensor(empty).counts.tolist() == [0, 0]

    with pytest.raises(ValueError, match="one-dimensional"):
        EdgeSelection(TensorBatch(np.ones((2, 1)), counts=[2]))
