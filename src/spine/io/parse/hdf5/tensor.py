"""Lightweight parsers for cached HDF5 tensor products."""

from __future__ import annotations

import numpy as np

from ..base import ParserBase
from ..data import ParserTensor

__all__ = ["HDF5FeatureTensorParser"]


class HDF5FeatureTensorParser(ParserBase):
    """Build a feature-only :class:`ParserTensor` from a cached HDF5 array."""

    name = "feature_tensor"
    aliases = ("tensor",)
    returns = "tensor"

    def __call__(self, trees):
        """Parse one cached entry."""
        return self.process(**self.get_input_data(trees))

    def process(self, tensor_event):
        """Cast a cached per-entry array into a feature-only parser tensor."""
        features = np.asarray(tensor_event, dtype=self.ftype)
        return ParserTensor(features=features, feats_only=True)
