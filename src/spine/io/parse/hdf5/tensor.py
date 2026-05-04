"""Lightweight parsers for cached HDF5 tensor products."""

from __future__ import annotations

import numpy as np

from ..base import ParserBase
from ..data import ParserTensor

__all__ = ["HDF5FeatureTensorParser", "HDF5IndexListParser"]


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


class HDF5IndexListParser(ParserBase):
    """Build an index-list :class:`ParserTensor` from cached HDF5 data."""

    name = "index_list"
    returns = "tensor"

    def __call__(self, trees):
        """Parse one cached entry."""
        return self.process(**self.get_input_data(trees))

    def process(self, index_event, count_event=None):
        """Normalize cached lists of indexes for collation into an IndexBatch."""
        index_list = []
        for index in index_event:
            index_list.append(np.asarray(index, dtype=self.itype).reshape(-1))

        single_counts = np.asarray(
            [len(index) for index in index_list], dtype=self.itype
        )
        global_shift = self.resolve_global_shift(index_list, count_event)

        return ParserTensor(
            features=index_list,
            global_shift=global_shift,
            single_counts=single_counts,
        )

    def resolve_global_shift(
        self,
        index_list: list[np.ndarray],
        count_event=None,
    ) -> int:
        """Determine the offset range spanned by one cached entry."""
        if count_event is not None:
            count_array = np.asarray(count_event)
            if count_array.ndim == 0:
                return int(count_array)
            return int(len(count_array))

        if not index_list:
            return 0

        full_index = (
            np.concatenate(index_list) if len(index_list) > 1 else index_list[0]
        )
        return int(np.max(full_index, initial=-1) + 1)
