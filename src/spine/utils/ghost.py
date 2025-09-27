"""Algorithms associated with the deghosting process."""

import numpy as np
import torch

from spine.data import TensorBatch

from .globals import DELTA_SHP, MICHL_SHP, SHOWR_SHP, TRACK_SHP


class ChargeRescaler:
    """Rescales the space point charge based on the deghosting output.

    It ensures that the amount of charge carried by each hit that makes up at
    least one space point is not duplicated by distributing said hit charge
    across all the space points formed with it.
    """

    def __init__(self, collection_only=False, collection_id=2):
        """Initialize the charge rescaler.

        Parameters
        ----------
        collection_only : bool, default False
            If `True`, only use the collection plane to estimate the rescaled charge
        collection_id : int, default 2
            Index of the collection plane
        """
        # Save the parameters
        self.collection_only = collection_only
        self.collection_id = collection_id

    def __call__(self, data):
        """Rescale the charge of one batch of deghosted data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f + 6) tensor of voxel/value pairs

        Returns
        -------
        data : Union[np.ndarray, torch.Tensor]
            (N) Rescaled charge values
        """
        charges = data._empty(len(data.tensor))
        for b in range(data.batch_size):
            lower, upper = data.edges[b], data.edges[b + 1]
            charges[lower:upper] = self.process_single(data[b])

        return charges

    def process_single(self, data):
        """Rescale the charge of one event.

        The last 6 columns of the input tensor *MUST* contain:
        - charge in each of the projection planes (3)
        - unique index of the hit in each 2D projection (3)

        Notes
        -----
        This function should work on numpy arrays or Torch tensors.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            (N, 1 + D + N_f + 6) tensor of voxel/value pairs

        Returns
        -------
        data : Union[np.ndarray, torch.Tensor]
            (N) Rescaled charge values
        """
        # Define operations on the basis of the input type
        if torch.is_tensor(data):
            unique, where = torch.unique, torch.where
            sum = lambda x: torch.sum(x, dim=1)
        else:
            unique, where = np.unique, np.where
            sum = lambda x: np.sum(x, axis=1)

        # Count how many times each wire hit is used to form a space point
        hit_ids = data[:, -3:]
        _, inverse, counts = unique(hit_ids, return_inverse=True, return_counts=True)
        mult = counts[inverse].reshape(-1, 3)

        # Rescale the charge on the basis of hit multiplicity
        hit_charges = data[:, -6:-3]
        if not self.collection_only:
            # Take the average of the charge estimates from each active plane
            pmask = hit_ids > -1
            charges = sum((hit_charges * pmask) / mult) / sum(pmask)
        else:
            # Only use the collection plane measurement, when available
            charges = hit_charges[:, self.collection_id] / mult[:, self.collection_id]

            # Fallback on the average if there is no collection hit
            bad_index = where(hit_ids[:, self.collection_id] < 0)[0]
            if len(bad_index) > 0:
                pmask = hit_ids[bad_index] > -1
                charges[bad_index] = sum(
                    (hit_charges[bad_index] * pmask) / mult[bad_index]
                ) / sum(pmask)

        return charges
