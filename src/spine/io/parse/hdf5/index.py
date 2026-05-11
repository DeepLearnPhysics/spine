"""Lightweight parsers for cached HDF5 index products."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import ParserBase
from ..data import ParserTensor

__all__ = ["HDF5IndexListParser", "HDF5EdgeIndexParser"]


class HDF5IndexListParser(ParserBase):
    """Build an index-list :class:`ParserTensor` from cached HDF5 data."""

    name = "index_list"
    returns = "tensor"

    def __call__(self, trees: dict[str, Any]) -> ParserTensor:
        """Parse one cached entry into a jagged index-list parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        ParserTensor
            Parser tensor containing a list of 1D index arrays and their
            batching metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self, index_event: np.ndarray, count_event: np.ndarray | None = None
    ) -> ParserTensor:
        """Normalize cached index lists for collation into an :class:`IndexBatch`.

        Parameters
        ----------
        index_event : np.ndarray
            Object array or nested array containing one index list per element.
        count_event : np.ndarray, optional
            Cached tensor used to infer the offset span of the entry.

        Returns
        -------
        ParserTensor
            Parser tensor containing normalized 1D index arrays.
        """
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
        count_event: np.ndarray | None = None,
    ) -> int:
        """Determine the offset span covered by one cached entry.

        Parameters
        ----------
        index_list : list[np.ndarray]
            Normalized 1D index arrays for one cached entry.
        count_event : np.ndarray, optional
            Cached tensor or scalar count used to infer the span directly.

        Returns
        -------
        int
            Global index span associated with the cached entry.
        """
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


class HDF5EdgeIndexParser(HDF5IndexListParser):
    """Build an edge-index :class:`ParserTensor` from cached HDF5 data."""

    name = "edge_index"

    def process(
        self, index_event: np.ndarray, count_event: np.ndarray | None = None
    ) -> ParserTensor:
        """Normalize cached edge indexes for collation into an EdgeIndexBatch.

        Parameters
        ----------
        index_event : np.ndarray
            Cached edge-index array with shape ``(2, E)`` or ``(E, 2)``.
        count_event : np.ndarray, optional
            Cached tensor used to infer the node span of the entry.

        Returns
        -------
        ParserTensor
            Parser tensor containing a normalized ``(2, E)`` edge-index array.
        """
        index = np.asarray(index_event, dtype=self.itype)
        if index.ndim != 2:
            raise ValueError(
                "Cached edge indexes must be 2D. "
                f"Received an array with shape {index.shape}."
            )

        if index.shape[0] != 2 and index.shape[1] == 2:
            index = index.T
        elif index.shape[0] != 2:
            raise ValueError(
                "Cached edge indexes must have shape (2, E) or (E, 2). "
                f"Received {index.shape}."
            )

        global_shift = self.resolve_global_shift([], count_event)
        return ParserTensor(features=index, global_shift=global_shift)
