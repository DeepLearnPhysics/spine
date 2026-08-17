"""Applies random smearing to deposition values."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["SmearingCalibrator"]


class SmearingCalibrator:
    """Smears deposition values with samples from a random distribution."""

    name = "smearing"

    def __init__(
        self,
        scale: float,
        distribution: str = "normal",
        mode: str = "additive",
        mean: float = 0.0,
        scope: str = "voxel",
        clip_min: float | None = None,
    ) -> None:
        """Initialize the smearing model.

        Parameters
        ----------
        scale : float
            Standard deviation of the sampled values. Must be nonnegative.
        distribution : str, default 'normal'
            Distribution to sample from. Currently only 'normal' is supported.
        mode : str, default 'additive'
            How to apply each sample to the corresponding deposition. 'additive'
            evaluates ``x + sample``, while 'multiplicative' evaluates
            ``x * sample``.
        mean : float, default 0.0
            Mean of the sampled values.
        scope : str, default 'voxel'
            Scope over which samples are shared. 'voxel' draws one sample per
            deposition, while 'image' draws one sample for the full image.
        clip_min : float, optional
            If specified, clip the smeared values to this lower bound.
        """
        if distribution != "normal":
            raise ValueError(
                f"Smearing distribution not recognized: {distribution}. "
                "Must be 'normal'."
            )
        if mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"Smearing mode not recognized: {mode}. Must be one of "
                "'additive' or 'multiplicative'."
            )
        if scope not in ("voxel", "image"):
            raise ValueError(
                f"Smearing scope not recognized: {scope}. Must be one of "
                "'voxel' or 'image'."
            )
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("Smearing scale must be a finite, nonnegative value.")
        if not np.isfinite(mean):
            raise ValueError("Smearing mean must be finite.")
        if clip_min is not None and not np.isfinite(clip_min):
            raise ValueError("Smearing lower bound must be finite.")

        self.distribution = distribution
        self.mode = mode
        self.mean = mean
        self.scale = scale
        self.scope = scope
        self.clip_min = clip_min

    def sample(
        self, size: tuple[int, ...] | None = None
    ) -> float | NDArray[np.floating]:
        """Draw samples for a deposition array.

        Parameters
        ----------
        size : tuple, optional
            Shape of the deposition array. Used for voxel-level smearing and
            ignored for image-level smearing.

        Returns
        -------
        float or np.ndarray
            One image-level sample or an array of voxel-level samples.
        """
        sample_size = size if self.scope == "voxel" else None
        return np.random.normal(loc=self.mean, scale=self.scale, size=sample_size)

    def process(
        self,
        values: NDArray[np.floating],
        sample: float | NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Apply random smearing to deposition values.

        Parameters
        ----------
        values : np.ndarray
            (N) array of deposition values
        sample : float or np.ndarray, optional
            Pre-sampled value or values. Used to share an image-level sample
            across TPC partitions.

        Returns
        -------
        np.ndarray
            (N) array of smeared deposition values
        """
        if sample is None:
            sample = self.sample(values.shape)

        if self.mode == "additive":
            result = values + sample
        else:
            result = values * sample

        if self.clip_min is not None:
            result = np.maximum(result, self.clip_min)

        return np.asarray(result, dtype=values.dtype)
