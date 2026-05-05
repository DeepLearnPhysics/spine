"""Lightweight parsers for cached HDF5 tensor products."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import ParserBase
from ..data import ParserTensor

__all__ = ["HDF5FeatureTensorParser"]


class HDF5FeatureTensorParser(ParserBase):
    """Build a feature-only :class:`ParserTensor` from a cached HDF5 array."""

    name = "feature_tensor"
    aliases = ("tensor",)
    returns = "tensor"

    def __init__(
        self,
        dtype: str,
        feature_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the cached feature-tensor parser.

        Parameters
        ----------
        dtype : str
            Floating-point dtype used by parser outputs.
        feature_cols : sequence[int], optional
            Optional list of feature-column indices to keep from the cached
            tensor. When provided, this acts as a feature ablation step before
            the parser tensor is returned.
        **kwargs : dict, optional
            Parser configuration forwarded to :class:`ParserBase`.
        """
        super().__init__(dtype, **kwargs)
        self.feature_cols = None
        if feature_cols is not None:
            self.feature_cols = np.asarray(feature_cols, dtype=np.int64)

    def __call__(self, trees: dict[str, Any]) -> ParserTensor:
        """Parse one cached entry into a feature-only parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        ParserTensor
            Feature-only parser tensor built from the cached array.
        """
        return self.process(**self.get_input_data(trees))

    def process(self, tensor_event: np.ndarray) -> ParserTensor:
        """Cast one cached per-entry array into a feature-only parser tensor.

        Parameters
        ----------
        tensor_event : np.ndarray
            Cached feature array for one event entry.

        Returns
        -------
        ParserTensor
            Feature-only parser tensor with ``features`` cast to the parser
            float dtype.
        """
        features = np.asarray(tensor_event, dtype=self.ftype)
        if self.feature_cols is not None:
            if features.ndim != 2:
                raise ValueError(
                    "Feature ablation requires a 2D cached feature tensor. "
                    f"Received an array with shape {features.shape}."
                )
            features = features[:, self.feature_cols]

        return ParserTensor(features=features, feats_only=True)
