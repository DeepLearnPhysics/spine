"""Tests for shared model-component factories."""

import pytest
import torch

from spine.model.common.factories import loss_fn_factory


def test_functional_binary_cross_entropy_variants_are_distinct():
    """BCE and BCE-with-logits names resolve to their respective functions."""
    bce = loss_fn_factory("bce", functional=True)
    bce_logits = loss_fn_factory("bce_logits", functional=True)

    assert bce is torch.nn.functional.binary_cross_entropy
    assert bce_logits is torch.nn.functional.binary_cross_entropy_with_logits


def test_loss_factory_constructs_modules_and_validates_functionals():
    """Module and functional paths enforce their distinct config contracts."""
    assert isinstance(loss_fn_factory("mse"), torch.nn.MSELoss)

    with pytest.raises(ValueError, match="only provide"):
        loss_fn_factory({"name": "ce", "reduction": "sum"}, functional=True)
    with pytest.raises(KeyError, match="Could not find"):
        loss_fn_factory("not_a_loss", functional=True)
