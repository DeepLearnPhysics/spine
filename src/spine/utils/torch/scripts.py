"""PyTorch utility scripts and extensions for enhanced functionality.

This module provides local extensions and utility functions that enhance
or work around limitations in PyTorch's built-in functionality.
"""

from ..conditional import TORCH_AVAILABLE, torch

__all__ = ["cdist_fast"]


def cdist_fast(v1, v2, metric="euclidean"):
    """Computes the pairwise distances between two `torch.Tensor` objects.

    This is necessary because the torch.cdist implementation is either
    slower (with the `donot_use_mm_for_euclid_dist` option) or produces
    dramatically wrong answers under certain situations (with the
    `use_mm_for_euclid_dist_if_necessary` option).

    Parameters
    ----------
    v1 : torch.Tensor
        (N, D) tensor of coordinates
    v2 : torch.Tensor
        (M, D) tensor of coordinates
    metric : str
        Distance metric

    Returns
    -------
    torch.Tensor
        (N, M) tensor of pairwise distances
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for cdist_fast functionality. "
            "Install with: pip install spine-ml[model]"
        )

    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1))
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1))
    if metric == "cityblock":
        return torch.abs(v2_2 - v1_2).sum(2)
    elif metric == "euclidean":
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))
    elif metric == "chebyshev":
        return torch.abs(v2_2 - v1_2).amax(2)
