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
        mode: str = "relative",
        mean: float = 0.0,
        clip_min: float | None = None,
    ) -> None:
        """Initialize the smearing model.

        Parameters
        ----------
        scale : float
            Standard deviation of the sampled values. Must be nonnegative.
        distribution : str, default 'normal'
            Distribution to sample from. Currently only 'normal' is supported.
        mode : str, default 'relative'
            How to apply each sample to the corresponding deposition. 'relative'
            evaluates ``x * (1 + sample)``, while 'additive' evaluates
            ``x + sample``.
        mean : float, default 0.0
            Mean of the sampled values.
        clip_min : float, optional
            If specified, clip the smeared values to this lower bound.
        """
        if distribution != "normal":
            raise ValueError(
                f"Smearing distribution not recognized: {distribution}. "
                "Must be 'normal'."
            )
        if mode not in ("relative", "additive"):
            raise ValueError(
                f"Smearing mode not recognized: {mode}. Must be one of "
                "'relative' or 'additive'."
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
        self.clip_min = clip_min

    def process(self, values: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply random smearing to deposition values.

        Parameters
        ----------
        values : np.ndarray
            (N) array of deposition values

        Returns
        -------
        np.ndarray
            (N) array of smeared deposition values
        """
        samples = np.random.normal(loc=self.mean, scale=self.scale, size=values.shape)
        if self.mode == "relative":
            result = values * (1.0 + samples)
        else:
            result = values + samples

        if self.clip_min is not None:
            result = np.maximum(result, self.clip_min)

        return np.asarray(result, dtype=values.dtype)
