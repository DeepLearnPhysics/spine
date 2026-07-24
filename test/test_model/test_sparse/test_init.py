"""Tests for the public sparse package facade."""

import pytest
import torch

from spine.model import sparse


def test_cat_concatenates_features_and_preserves_provenance():
    """Sparse concatenation operates on a shared coordinate map."""
    first = sparse.SparseTensor(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32),
        batch_size=2,
    )
    second = first.replace_features(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))

    output = sparse.cat([first, second])

    assert torch.equal(output.C, first.C)
    assert torch.equal(
        output.F,
        torch.tensor([[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]),
    )
    assert output.batch_size == 2


def test_cat_handles_empty_tensors_without_entering_backend():
    """Empty concatenation preserves metadata and sums channel counts."""
    first = sparse.SparseTensor(
        torch.empty((0, 2)),
        torch.empty((0, 4), dtype=torch.int32),
        batch_size=3,
    )
    second = sparse.SparseTensor(
        torch.empty((0, 4)),
        torch.empty((0, 4), dtype=torch.int32),
        batch_size=3,
    )

    output = sparse.cat(first, second)

    assert output.shape == (0, 6)
    assert output.counts.tolist() == [0, 0, 0]


def test_cat_forwards_native_tensors_to_backend():
    """Native backend inputs remain supported at the adapter boundary."""
    first = sparse.SparseTensor(
        torch.tensor([[1.0]]),
        torch.tensor([[0, 0, 0]], dtype=torch.int32),
    )
    second = first.replace_features(torch.tensor([[2.0]]))

    output = sparse.cat(first.backend_tensor, second.backend_tensor)

    assert torch.equal(output.F, torch.tensor([[1.0, 2.0]]))


def test_cat_requires_an_input():
    """Sparse concatenation rejects an empty argument list."""
    with pytest.raises(ValueError, match="at least one tensor"):
        sparse.cat()
