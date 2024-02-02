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


def to_numpy(array):
    """Casts an array-like objecs to a `np.ndarray`.

    Parameters
    ----------
    array : object
        Array-like object (can be either `np.ndarray`, `torch.Tensor`
        or `ME.SparseTensor`)

    Returns
    -------
    np.ndarray
        Array cast to np.ndarray
    """
    import MinkowskiEngine as ME

    if isinstance(array, (list, tuple)):
        return np.array(array)
    elif isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        if array.ndim == 0:
            return array.item()
        else:
            return array.cpu().detach().numpy()
    elif isinstance(array, ME.SparseTensor):
        tensor = torch.cat([array.C.float(), array.F], dim=1)
        return tensor.detach().cpu().numpy()
    else:
        raise TypeError(f'Unknown return type: {type(type)}')


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
