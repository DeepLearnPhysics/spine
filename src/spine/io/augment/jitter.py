"""Voxel coordinate jitter augmentation module."""

from typing import Any

import numpy as np

from spine.data import Meta

from .base import AugmentBase


class JitterAugment(AugmentBase):
    """Generic class to handle voxel coordinate jitter."""

    name = "jitter"

    def __init__(
        self,
        max_offset: int | tuple[int, int, int] | list[int] | np.ndarray,
        distribution: str = "uniform",
        poisson_lambda: (
            float | tuple[float, float, float] | list[float] | np.ndarray
        ) = 1.0,
        clip: bool = True,
    ) -> None:
        """Initialize the jitter augmenter.

        Parameters
        ----------
        max_offset : Union[int, Tuple[int, int, int], List[int], np.ndarray]
            Maximum absolute voxel shift to apply along each axis. If an integer
            is provided, the same bound is used for all three axes.
        distribution : str, default "uniform"
            Offset distribution to sample from. Supported values are
            `"uniform"` and `"poisson"`.
        poisson_lambda : Union[float, Tuple[float, float, float], List[float], np.ndarray], default 1.0
            Mean of the Poisson distribution used when
            `distribution == "poisson"`. If a scalar is provided, the same
            value is used for all three axes.
        clip : bool, default True
            If `True`, clamp jittered coordinates to the image bounds.

        Returns
        -------
        None
            This method does not return anything
        """
        if np.isscalar(max_offset):
            assert not isinstance(max_offset, complex)  # Type narrowing for mypy
            max_offset = np.full(3, int(max_offset), dtype=np.int64)
        else:
            max_offset = np.asarray(max_offset, dtype=np.int64)

        if len(max_offset) != 3:
            raise ValueError("Must provide a jitter bound for each axis.")
        if np.any(max_offset < 0):
            raise ValueError("Jitter bounds must be non-negative.")
        if distribution not in ("uniform", "poisson"):
            raise ValueError(
                "Jitter distribution not recognized. Must be one of "
                "('uniform', 'poisson')."
            )

        if np.isscalar(poisson_lambda):
            assert not isinstance(poisson_lambda, complex)  # Type narrowing for mypy
            poisson_lambda = np.full(3, float(poisson_lambda), dtype=np.float32)
        else:
            poisson_lambda = np.asarray(poisson_lambda, dtype=np.float32)

        if len(poisson_lambda) != 3:
            raise ValueError("Must provide a Poisson mean for each axis.")
        if np.any(poisson_lambda < 0.0):
            raise ValueError("Poisson means must be non-negative.")

        self.max_offset = max_offset
        self.distribution = distribution
        self.poisson_lambda = poisson_lambda
        self.clip = clip

    def apply(
        self,
        data: dict[str, Any],
        meta: Meta,
        keys: list[str],
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], Meta]:
        """Apply per-voxel coordinate jitter.

        Parameters
        ----------
        data : dict
            Dictionary of event data products to augment
        meta : Meta
            Shared image metadata
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        context : dict
            Shared augmentation context

        Returns
        -------
        Tuple[Dict[str, Any], Meta]
            Updated data dictionary and unchanged metadata
        """
        for key in keys:
            if isinstance(data[key], Meta):
                continue

            coords = data[key].coords
            offsets = self.generate_offsets(len(coords))
            coords = coords + offsets

            if self.clip:
                coords = np.clip(coords, 0, meta.count - 1)

            data[key].coords = coords.astype(data[key].coords.dtype)
            data[key].meta = meta

        return data, meta

    def generate_offsets(self, num_voxels: int) -> np.ndarray:
        """Generate random integer voxel offsets.

        Parameters
        ----------
        num_voxels : int
            Number of voxel coordinates to jitter

        Returns
        -------
        np.ndarray
            Integer per-voxel offsets of shape `(num_voxels, 3)`
        """
        if self.distribution == "uniform":
            return np.random.randint(
                -self.max_offset, self.max_offset + 1, size=(num_voxels, 3)
            )

        return self.generate_poisson_offsets(num_voxels)

    def generate_poisson_offsets(self, num_voxels: int) -> np.ndarray:
        """Generate signed Poisson-distributed voxel offsets.

        Parameters
        ----------
        num_voxels : int
            Number of voxel coordinates to jitter

        Returns
        -------
        np.ndarray
            Integer per-voxel offsets of shape `(num_voxels, 3)`
        """
        magnitudes = np.random.poisson(
            self.poisson_lambda, size=(num_voxels, 3)
        ).astype(np.int64)
        magnitudes = np.minimum(magnitudes, self.max_offset)

        signs = np.random.randint(0, 2, size=(num_voxels, 3), dtype=np.int64)
        signs = 2 * signs - 1

        return signs * magnitudes
