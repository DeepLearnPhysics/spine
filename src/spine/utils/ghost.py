"""Algorithms associated with the deghosting process."""

from typing import Any

import numpy as np

from spine.data import TensorBatch
from spine.utils.conditional import torch


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

    def __call__(self, data: TensorBatch, return_info: bool = False) -> Any:
        """Rescale the charge of one batch of deghosted data.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f + 6) tensor of voxel/value pairs
        return_info : bool, default False
            Return the three input plane charges and their hit multiplicities
            alongside the rescaled charge

        Returns
        -------
        Union[np.ndarray, torch.Tensor, tuple]
            (N) rescaled charge values. If ``return_info`` is ``True``, also
            returns (N, 3) plane-charge and hit-multiplicity arrays.
        """
        # Charge must retain the input floating-point dtype. ``TensorBatch``'s
        # generic ``_empty`` helper intentionally allocates integer indexes.
        charges: Any
        plane_charges: Any = None
        multiplicities: Any = None
        if torch.is_tensor(data.tensor):
            charges = torch.empty(
                len(data.tensor), dtype=data.dtype, device=data.device
            )
            if return_info:
                plane_charges = torch.empty(
                    (len(charges), 3), dtype=data.dtype, device=data.device
                )
                multiplicities = torch.empty(
                    (len(charges), 3), dtype=torch.long, device=data.device
                )
        else:
            charges = np.empty(len(data.tensor), dtype=data.dtype)
            if return_info:
                plane_charges = np.empty((len(charges), 3), dtype=data.dtype)
                multiplicities = np.empty((len(charges), 3), dtype=np.int64)

        for b in range(data.batch_size):
            lower, upper = data.edges[b], data.edges[b + 1]
            result = self.process_single(data[b], return_info=return_info)
            if return_info:
                assert plane_charges is not None
                assert multiplicities is not None
                charge, plane_charge, multiplicity = result
                charges[lower:upper] = charge
                plane_charges[lower:upper] = plane_charge
                multiplicities[lower:upper] = multiplicity
            else:
                charges[lower:upper] = result

        if return_info:
            assert plane_charges is not None
            assert multiplicities is not None
            return charges, plane_charges, multiplicities
        return charges

    def process_single(self, data: Any, return_info: bool = False) -> Any:
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
        return_info : bool, default False
            Return the per-plane inputs used to form the rescaled charge

        Returns
        -------
        Union[np.ndarray, torch.Tensor, tuple]
            (N) rescaled charge values. If ``return_info`` is ``True``, also
            returns the (N, 3) plane charges and corresponding multiplicities.
        """
        # Define operations on the basis of the input type
        if torch.is_tensor(data):
            unique, where = torch.unique, torch.where
        else:
            unique, where = np.unique, np.where

        # Count how many times each wire hit is used to form a space point
        hit_ids = data[:, -3:]
        _, inverse, counts = unique(hit_ids, return_inverse=True, return_counts=True)
        mult = counts[inverse].reshape(-1, 3)

        # Rescale the charge on the basis of hit multiplicity
        hit_charges = data[:, -6:-3]
        if not self.collection_only:
            # Take the average of the charge estimates from each active plane
            pmask = hit_ids > -1
            charges = self._sum_rows((hit_charges * pmask) / mult) / self._sum_rows(
                pmask
            )
        else:
            # Only use the collection plane measurement, when available
            charges = hit_charges[:, self.collection_id] / mult[:, self.collection_id]

            # Fallback on the average if there is no collection hit
            bad_index = where(hit_ids[:, self.collection_id] < 0)[0]
            if len(bad_index) > 0:
                pmask = hit_ids[bad_index] > -1
                charges[bad_index] = self._sum_rows(
                    (hit_charges[bad_index] * pmask) / mult[bad_index]
                ) / self._sum_rows(pmask)

        if return_info:
            # Missing-plane sentinels are grouped by ``unique`` for the
            # arithmetic above but do not represent a used hit. Report zero
            # so the diagnostic is the physical use count for every plane.
            info_mult = torch.clone(mult) if torch.is_tensor(mult) else np.copy(mult)
            info_mult[hit_ids < 0] = 0
            return charges, hit_charges, info_mult

        return charges

    @staticmethod
    def _sum_rows(values: Any) -> Any:
        """Sum matrix rows without changing the active array backend."""
        if torch.is_tensor(values):
            return torch.sum(values, dim=1)
        return np.sum(values, axis=1)
