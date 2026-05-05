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
        return ParserTensor(features=features, feats_only=True)
