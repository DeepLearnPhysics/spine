"""Tests for class-weight construction."""

import numpy as np
import pytest
import torch

from spine.model.common.weighting import get_class_weights


@pytest.mark.parametrize("mode", ["const", "log", "sqrt"])
@pytest.mark.parametrize("per_class", [True, False])
def test_class_weights_numpy_and_torch(mode, per_class):
    """Every weighting mode should support class-wise and element-wise output."""
    labels = np.array([0, 0, 1], dtype=np.int64)
    numpy_weights = get_class_weights(labels, 3, mode, per_class)
    torch_weights = get_class_weights(torch.as_tensor(labels), 3, mode, per_class)
    expected_length = 3
    assert len(numpy_weights) == expected_length
    assert len(torch_weights) == expected_length
    np.testing.assert_allclose(torch_weights.cpu().numpy(), numpy_weights)


def test_class_weights_reject_unknown_mode():
    """Unknown weighting transformations should fail explicitly."""
    with pytest.raises(ValueError, match="not recognized"):
        get_class_weights(np.array([0]), 1, "bad")
