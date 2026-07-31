"""Tests for shared model-component factories."""

import torch

from spine.model.common.factories import loss_fn_factory


def test_functional_binary_cross_entropy_variants_are_distinct():
    """BCE and BCE-with-logits names resolve to their respective functions."""
    bce = loss_fn_factory("bce", functional=True)
    bce_logits = loss_fn_factory("bce_logits", functional=True)

    assert bce is torch.nn.functional.binary_cross_entropy
    assert bce_logits is torch.nn.functional.binary_cross_entropy_with_logits
