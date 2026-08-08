"""Lightweight parsers for cached HDF5 index products."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import EdgeIndexData, IndexData, IndexListData

from ..base import ParserBase
from .utils import resolve_index_span

__all__ = ["HDF5IndexParser", "HDF5IndexListParser", "HDF5EdgeIndexParser"]


class HDF5IndexParser(ParserBase):
    """Build a flat :class:`IndexData` from cached HDF5 data."""

    name = "index"

    def __init__(self, dtype: str, index_event: str, count_event: str) -> None:
        """Initialize the cached flat-index parser.

        Parameters
        ----------
        dtype : str
            Floating-point dtype shared with the parser configuration.
        index_event : str
            HDF5 product containing the event's flat indexes.
        count_event : str
            HDF5 product used to recover the indexed parent span.
        """
        super().__init__(
            dtype,
            index_event=index_event,
            count_event=count_event,
        )

    def __call__(self, trees: dict[str, Any]) -> IndexData:
        """Parse one cached entry into a flat index parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        IndexData
            Parser index containing one normalized 1D index array and its
            batching metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(self, index_event: np.ndarray, count_event: np.ndarray) -> IndexData:
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
        IndexData
            Parser index containing one normalized 1D index array.
        """
        # Normalize the physical index and recover its parent namespace size
        index = np.asarray(index_event, dtype=self.itype).reshape(-1)
        span = resolve_index_span(count_event)

        return IndexData(features=index, span=span)


class HDF5IndexListParser(ParserBase):
    """Build an index-list :class:`IndexListData` from cached HDF5 data."""

    name = "index_list"

    def __init__(self, dtype: str, index_event: str, count_event: str) -> None:
        """Initialize the cached index-list parser.

        Parameters
        ----------
        dtype : str
            Floating-point dtype shared with the parser configuration.
        index_event : str
            HDF5 product containing the event's jagged index collection.
        count_event : str
            HDF5 product used to recover the indexed parent span.
        """
        super().__init__(
            dtype,
            index_event=index_event,
            count_event=count_event,
        )

    def __call__(self, trees: dict[str, Any]) -> IndexListData:
        """Parse one cached entry into a jagged index-list parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        IndexListData
            Parser index list containing 1D index arrays and their batching
            metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self, index_event: np.ndarray, count_event: np.ndarray
    ) -> IndexListData:
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
        IndexListData
            Parser index list containing normalized 1D index arrays.
        """
        # Normalize each jagged member independently
        index_list = []
        for index in index_event:
            index_list.append(np.asarray(index, dtype=self.itype).reshape(-1))

        # Preserve both member boundaries and the parent namespace size
        single_counts = np.asarray(
            [len(index) for index in index_list], dtype=self.itype
        )
        span = resolve_index_span(count_event)

        return IndexListData(
            features=index_list,
            span=span,
            single_counts=single_counts,
        )


class HDF5EdgeIndexParser(ParserBase):
    """Build an edge-index :class:`EdgeIndexData` from cached HDF5 data."""

    name = "edge_index"

    def __init__(self, dtype: str, index_event: str, count_event: str) -> None:
        """Initialize the cached edge-index parser.

        Parameters
        ----------
        dtype : str
            Floating-point dtype shared with the parser configuration.
        index_event : str
            HDF5 product containing the event's edge-index matrix.
        count_event : str
            HDF5 product used to recover the indexed node span.
        """
        super().__init__(
            dtype,
            index_event=index_event,
            count_event=count_event,
        )

    def __call__(self, trees: dict[str, Any]) -> EdgeIndexData:
        """Parse one cached entry into an edge-index parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        EdgeIndexData
            Parser edge index containing a normalized 2D edge array and its
            batching metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self, index_event: np.ndarray, count_event: np.ndarray
    ) -> EdgeIndexData:
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
        EdgeIndexData
            Parser edge index containing a normalized ``(2, E)`` array.
        """
        # Validate the matrix before normalizing its orientation
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

        # Recover the node namespace required for batched index shifting
        span = resolve_index_span(count_event)

        return EdgeIndexData(features=index, span=span)
