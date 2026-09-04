"""Detector-response augmentation for sparse tensor features."""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from spine.data import Meta, TensorData

from .base import AugmentBase


class ResponseAugment(AugmentBase):
    """Perturb explicitly selected detector-response feature columns.

    The augmenter preserves coordinates and row counts so that truth products
    and other row-aligned inputs remain valid. Thresholding and stochastic
    signal loss replace affected values with ``fill_value`` rather than
    deleting sparse rows.
    """

    name = "response"

    def __init__(
        self,
        features: Mapping[str, int | Sequence[int]],
        gain_range: Sequence[float] = (1.0, 1.0),
        noise_sigma: float = 0.0,
        dropout_prob: float = 0.0,
        threshold: float | None = None,
        saturation: float | None = None,
        clip_min: float | None = None,
        fill_value: float = 0.0,
        p: float = 1.0,
    ) -> None:
        """Initialize detector-response perturbations.

        Parameters
        ----------
        features : mapping[str, int or sequence[int]]
            Mapping from tensor-product names to feature-relative columns to
            modify. Explicit product selection prevents response changes from
            being applied to semantic or instance-label tensors accidentally.
        gain_range : sequence[float], default (1.0, 1.0)
            Inclusive range from which one event-level multiplicative gain is
            sampled. The same gain is applied to every configured product.
        noise_sigma : float, default 0.0
            Standard deviation of independent additive Gaussian noise.
        dropout_prob : float, default 0.0
            Probability of replacing all selected response columns for one
            sparse row with ``fill_value``. Rows at identical coordinates use
            the same dropout decision across configured products.
        threshold : float, optional
            Replace perturbed feature values below this threshold with
            ``fill_value``. This models zero suppression while retaining row
            alignment.
        saturation : float, optional
            Upper value at which perturbed responses are clipped.
        clip_min : float, optional
            Lower value at which perturbed responses are clipped.
        fill_value : float, default 0.0
            Replacement used by thresholding and stochastic signal loss.
        p : float, default 1.0
            Event-level probability of applying this augmentation.
        """
        if not isinstance(features, Mapping) or not features:
            raise ValueError("Response augmentation requires a feature mapping.")

        # Normalize feature selections once so event processing stays simple.
        self.features: dict[str, np.ndarray] = {}
        for key, columns in features.items():
            if not isinstance(key, str):
                raise TypeError("Response product names must be strings.")
            if isinstance(columns, Integral) and not isinstance(columns, bool):
                columns = (int(columns),)
            try:
                raw_array = np.asarray(columns)
            except (TypeError, ValueError) as err:
                raise TypeError(
                    f"Response feature columns for `{key}` must be integers."
                ) from err
            if raw_array.ndim != 1 or len(raw_array) == 0:
                raise ValueError(
                    f"Response feature columns for `{key}` must be a nonempty "
                    "one-dimensional sequence."
                )
            if not np.issubdtype(raw_array.dtype, np.integer) or np.issubdtype(
                raw_array.dtype, np.bool_
            ):
                raise TypeError(
                    f"Response feature columns for `{key}` must be integers."
                )
            array = raw_array.astype(np.int64, copy=False)
            if np.any(array < 0) or len(np.unique(array)) != len(array):
                raise ValueError(
                    f"Response feature columns for `{key}` must be unique and "
                    "non-negative."
                )
            self.features[key] = array

        gain = np.asarray(gain_range, dtype=np.float64)
        if gain.shape != (2,) or not np.all(np.isfinite(gain)):
            raise ValueError("`gain_range` must contain two finite values.")
        if gain[0] < 0.0 or gain[0] > gain[1]:
            raise ValueError("`gain_range` must satisfy 0 <= low <= high.")

        self.noise_sigma = self._nonnegative(noise_sigma, "noise_sigma")
        self.dropout_prob = self._probability(dropout_prob, "dropout_prob")
        self.p = self._probability(p, "p")
        self.gain_range = gain
        self.threshold = self._optional_finite(threshold, "threshold")
        self.saturation = self._optional_finite(saturation, "saturation")
        self.clip_min = self._optional_finite(clip_min, "clip_min")
        self.fill_value = self._finite(fill_value, "fill_value")
        if (
            self.clip_min is not None
            and self.saturation is not None
            and self.clip_min > self.saturation
        ):
            raise ValueError("`clip_min` cannot exceed `saturation`.")

    def apply(
        self,
        data: dict[str, Any],
        meta: Meta,
        keys: list[str],
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], Meta]:
        """Apply response perturbations to one event.

        Parameters
        ----------
        data : dict
            Event products containing every configured tensor.
        meta : Meta
            Shared spatial metadata, returned unchanged.
        keys : list[str]
            Coordinate-bearing products discovered by the augmentation manager.
            Selection is controlled by :attr:`features`, not this full list.
        context : dict
            Shared augmentation context. It is not modified by this module.

        Returns
        -------
        tuple[dict, Meta]
            Mutated event products and their unchanged spatial metadata.
        """
        if np.random.rand() >= self.p:
            return data, meta

        gain = np.random.uniform(*self.gain_range)
        dropped_coordinates: dict[tuple[int, ...], bool] = {}
        for key, columns in self.features.items():
            if key not in data:
                raise KeyError(f"Response product `{key}` is missing from the event.")
            product = data[key]
            if not isinstance(product, TensorData):
                raise TypeError(f"Response product `{key}` must be a TensorData.")

            values = np.asarray(product.features)
            one_dimensional = values.ndim == 1
            if one_dimensional:
                values = values.reshape(-1, 1)
            elif values.ndim != 2:
                raise ValueError(
                    f"Response product `{key}` must have one- or two-dimensional "
                    f"features, received shape {values.shape}."
                )
            if len(columns) and int(np.max(columns)) >= values.shape[1]:
                raise IndexError(
                    f"Response feature columns for `{key}` exceed its feature "
                    f"width of {values.shape[1]}."
                )

            # Work in floating point even when a cache stored integral ADCs,
            # then cast back into the product's established feature dtype.
            response = values[:, columns].astype(np.float64, copy=True)
            response *= gain
            if self.noise_sigma:
                response += np.random.normal(0.0, self.noise_sigma, response.shape)
            if self.clip_min is not None:
                np.maximum(response, self.clip_min, out=response)
            if self.saturation is not None:
                np.minimum(response, self.saturation, out=response)
            if self.threshold is not None:
                response[response < self.threshold] = self.fill_value

            # Signal-loss decisions are shared by physical coordinate when
            # multiple configured products contain the same sparse point.
            if self.dropout_prob:
                drop = self._dropout_mask(product, dropped_coordinates)
                response[drop] = self.fill_value

            values[:, columns] = response.astype(values.dtype, copy=False)
            product.features = values[:, 0] if one_dimensional else values

        return data, meta

    def _dropout_mask(
        self,
        product: TensorData,
        decisions: dict[tuple[int, ...], bool],
    ) -> np.ndarray:
        """Sample row loss, sharing decisions for repeated coordinates."""
        coords = product.coordinate_data
        if coords is None:
            return np.random.rand(len(product.features)) < self.dropout_prob

        mask = np.empty(len(coords), dtype=bool)
        for index, coord in enumerate(np.asarray(coords)):
            coordinate = tuple(int(value) for value in coord)
            if coordinate not in decisions:
                decisions[coordinate] = bool(np.random.rand() < self.dropout_prob)
            mask[index] = decisions[coordinate]
        return mask

    @staticmethod
    def _finite(value: float, name: str) -> float:
        """Normalize one required finite scalar parameter."""
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"`{name}` must be finite.")
        return value

    @classmethod
    def _optional_finite(cls, value: float | None, name: str) -> float | None:
        """Normalize one optional finite scalar parameter."""
        return None if value is None else cls._finite(value, name)

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        """Normalize one finite, non-negative scalar parameter."""
        value = cls._finite(value, name)
        if value < 0.0:
            raise ValueError(f"`{name}` must be non-negative.")
        return value

    @classmethod
    def _probability(cls, value: float, name: str) -> float:
        """Normalize one probability parameter."""
        value = cls._finite(value, name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"`{name}` must be in the range [0, 1].")
        return value
