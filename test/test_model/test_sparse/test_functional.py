"""Tests for sparse feature-wise functions."""

import torch

from spine.model import sparse


def test_softmax_preserves_coordinates_and_normalizes_features():
    """Sparse softmax changes features without changing active sites."""
    tensor = sparse.SparseTensor(
        torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
        torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.int32),
        batch_size=2,
    )

    output = sparse.softmax(tensor, dim=1)

    assert torch.equal(output.C, tensor.C)
    assert torch.allclose(output.F.sum(dim=1), torch.ones(2))
