"""Algorithms associated with the deghosting process."""

import numpy as np
import torch

from spine.data import TensorBatch

from .globals import SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP


def compute_rescaled_charge_batch(data, collection_only=False, collection_id=2):
    """Batched version of :func:`compute_rescaled_charge`.

    Parameters
    ----------
    data : TensorBatch
        (N, 1 + D + N_f + 6) tensor of voxel/value pairs
    collection_only : bool, default False
        Only use the collection plane to estimate the rescaled charge
    collection_id : int, default 2
        Index of the collection plane

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        (N) Rescaled charge values
    """
    charges = data._empty(len(data.tensor))
    for b in range(data.batch_size):
        lower, upper = data.edges[b], data.edges[b+1]
        charges[lower:upper] = compute_rescaled_charge(
                data[b], collection_only, collection_id)

    return charges


def compute_rescaled_charge(data, collection_only=False, collection_id=2):
    """Computes rescaled charge after deghosting.

    The last 6 columns of the input tensor *MUST* contain:
    - charge in each of the projective planes (3)
    - index of the hit in each 2D projection (3)

    Notes
    -----
    This function should work on numpy arrays or Torch tensors.

    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor]
        (N, 1 + D + N_f + 6) tensor of voxel/value pairs
    collection_only : bool, default False
        Only use the collection plane to estimate the rescaled charge
    collection_id : int, default 2
        Index of the collection plane

    Returns
    -------
    data : Union[np.ndarray, torch.Tensor]
        (N) Rescaled charge values
    """
    # Define operations on the basis of the input type
    if torch.is_tensor(data):
        unique = torch.unique
        empty = lambda shape: torch.empty(shape, dtype=torch.long,
                device=data.device)
        sum = lambda x: torch.sum(x, dim=1)
    else:
        unique = np.unique
        empty = np.empty
        sum = lambda x: np.sum(x, axis=1)

    # Count how many times each wire hit is used to form a space point
    hit_ids = data[:, -3:]
    _, inverse, counts = unique(
            hit_ids, return_inverse=True, return_counts=True)
    multiplicity = counts[inverse].reshape(-1, 3)

    # Rescale the charge on the basis of hit multiplicity
    hit_charges = data[:, -6:-3]
    if not collection_only:
        # Take the average of the charge estimates from each active plane
        pmask   = hit_ids > -1
        charges = sum((hit_charges*pmask)/multiplicity)/sum(pmask)
    else:
        # Only use the collection plane measurement
        charges = hit_charges[:, collection_id]/multiplicity[:, collection_id]

    return charges
