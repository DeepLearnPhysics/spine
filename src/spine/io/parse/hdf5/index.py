"""Lightweight parsers for cached HDF5 index products."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import ParserBase
from ..data import ParserEdgeIndex, ParserIndex, ParserIndexList
from .utils import resolve_index_global_shift

__all__ = ["HDF5IndexParser", "HDF5IndexListParser", "HDF5EdgeIndexParser"]


class HDF5IndexParser(ParserBase):
    """Build a flat :class:`ParserIndex` from cached HDF5 data."""

    name = "index"
    returns = "tensor"

    def __init__(self, dtype: str, index_event: str, count_event: str) -> None:
        """Require both the cached index and its parent-count hint."""
        super().__init__(
            dtype,
            index_event=index_event,
            count_event=count_event,
        )

    def __call__(self, trees: dict[str, Any]) -> ParserIndex:
        """Parse one cached entry into a flat index parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        ParserIndex
            Parser index containing one normalized 1D index array and its
            batching metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(self, index_event: np.ndarray, count_event: np.ndarray) -> ParserIndex:
        """Normalize one cached flat index for collation into an IndexBatch.

        Parameters
        ----------
        index_event : np.ndarray
            Cached flat index array for one event entry.
        count_event : np.ndarray
            Cached tensor or scalar count used to infer the offset span of the
            indexed parent entry.

        Returns
        -------
        ParserIndex
            Parser index containing one normalized 1D index array.
        """
        index = np.asarray(index_event, dtype=self.itype).reshape(-1)
        global_shift = resolve_index_global_shift(count_event)

        return ParserIndex(features=index, global_shift=global_shift)


class HDF5IndexListParser(ParserBase):
    """Build an index-list :class:`ParserIndexList` from cached HDF5 data."""

    name = "index_list"
    returns = "tensor"

    def __init__(self, dtype: str, index_event: str, count_event: str) -> None:
        """Require both the cached indexes and their parent-count hint."""
        super().__init__(
            dtype,
            index_event=index_event,
            count_event=count_event,
        )

    def __call__(self, trees: dict[str, Any]) -> ParserIndexList:
        """Parse one cached entry into a jagged index-list parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        ParserIndexList
            Parser index list containing 1D index arrays and their batching
            metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self, index_event: np.ndarray, count_event: np.ndarray
    ) -> ParserIndexList:
        """Normalize cached index lists for collation into an :class:`IndexBatch`.

        Parameters
        ----------
        index_event : np.ndarray
            Object array or nested array containing one index list per element.
        count_event : np.ndarray
            Cached tensor or scalar count used to infer the offset span of the
            indexed parent entry.

        Returns
        -------
        ParserIndexList
            Parser index list containing normalized 1D index arrays.
        """
        index_list = []
        for index in index_event:
            index_list.append(np.asarray(index, dtype=self.itype).reshape(-1))

        single_counts = np.asarray(
            [len(index) for index in index_list], dtype=self.itype
        )
        global_shift = resolve_index_global_shift(count_event)

        return ParserIndexList(
            features=index_list,
            global_shift=global_shift,
            single_counts=single_counts,
        )


class HDF5EdgeIndexParser(ParserBase):
    """Build an edge-index :class:`ParserEdgeIndex` from cached HDF5 data."""

    name = "edge_index"
    returns = "tensor"

    def __init__(self, dtype: str, index_event: str, count_event: str) -> None:
        """Require both the cached edge index and its parent-count hint."""
        super().__init__(
            dtype,
            index_event=index_event,
            count_event=count_event,
        )

    def __call__(self, trees: dict[str, Any]) -> ParserEdgeIndex:
        """Parse one cached entry into an edge-index parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        ParserEdgeIndex
            Parser edge index containing a normalized 2D edge array and its
            batching metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self, index_event: np.ndarray, count_event: np.ndarray
    ) -> ParserEdgeIndex:
        """Normalize cached edge indexes for collation into an EdgeIndexBatch.

        Parameters
        ----------
        index_event : np.ndarray
            Cached edge-index array with shape ``(2, E)`` or ``(E, 2)``.
        count_event : np.ndarray
            Cached tensor or scalar count used to infer the node span of the
            indexed parent entry.

        Returns
        -------
        ParserEdgeIndex
            Parser edge index containing a normalized ``(2, E)`` array.
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

        global_shift = resolve_index_global_shift(count_event)
        return ParserEdgeIndex(features=index, global_shift=global_shift)
