import numpy as np
import torch


def cycle(data_io):
    """Cycles over a torch DataLoader.

    Use this function instead of itertools.cycle to avoid creating a memory
    leak (itertools.cycle attempts to save all outputs in order to re-cycle
    through them)

    Parameters
    ----------
    data_io : torch.utils.data.DataLoader
        Data loader
    """
    while True:
        for x in data_io:
            yield x


def local_cdist(v1, v2):
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

    Returns
    -------
    torch.Tensor
        (N, M) tensor of pairwise distances
    """
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1))
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1))
    return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))


def unique_index(x, dim=None):
    """Returns the list of unique indexes in the tensor and their first index.

    This is a temporary implementation until PyTorch adds support for the
    `return_index` argument in their `torch.unique` function.

    Parameters
    ----------
    x : torch.Tensor
        (N) Tensor of values

    Returns
    -------
    unique : torch.Tensor
        (U) List of unique values in the input tensor
    index : torch.Tensor
        (U) List of the first index of each unique values
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])

    index = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    return unique.long(), index
            
