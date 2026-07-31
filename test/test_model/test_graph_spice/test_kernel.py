"""Behavioral tests for GraphSPICE edge kernels."""

import pytest
import torch

from spine.model.graph_spice.kernel import (
    BilinearKernel,
    DefaultKernel,
    MLPKernel,
)


def default_features(
    spatial: torch.Tensor,
    feature: torch.Tensor,
) -> torch.Tensor:
    """Build the feature contract consumed by ``DefaultKernel``."""
    count = len(spatial)
    covariance = torch.ones((count, 2))
    occupancy = torch.ones((count, 1))
    return torch.cat((spatial, feature, covariance, occupancy), dim=1)


def test_default_kernel_uses_both_endpoint_features():
    """Separated endpoints must score below identical endpoints."""
    kernel = DefaultKernel(num_features=2)
    first = default_features(torch.zeros((2, 3)), torch.zeros((2, 2)))
    second = default_features(torch.ones((2, 3)), torch.ones((2, 2)))

    identical_logits = kernel(first, first)
    separated_logits = kernel(first, second)

    assert torch.isfinite(identical_logits).all()
    assert torch.isfinite(separated_logits).all()
    assert torch.all(identical_logits > separated_logits)


@pytest.mark.parametrize(
    "kernel",
    [
        DefaultKernel(num_features=2),
        BilinearKernel(num_features=2),
        MLPKernel(num_features=2),
    ],
)
def test_kernels_validate_endpoint_shapes(kernel):
    """Kernels must reject mismatched endpoint tensors explicitly."""
    first = torch.zeros((3, 2))
    second = torch.zeros((2, 2))

    with pytest.raises(ValueError, match="matching endpoint tensors"):
        kernel(first, second)


@pytest.mark.parametrize("kernel_type", [BilinearKernel, MLPKernel])
def test_learned_kernels_return_one_logit_per_edge(kernel_type):
    """Learned kernels must preserve the input edge count."""
    kernel = kernel_type(num_features=4)
    first = torch.randn(5, 4)
    second = torch.randn(5, 4)

    logits = kernel(first, second)

    assert logits.shape == (5, 1)
