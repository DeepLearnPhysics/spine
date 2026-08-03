"""Batched graph edge-index data products.

An edge index is a sparse representation of a graph incidence matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..edge_index import EdgeIndexData
from .base import ArrayLike, BatchBase

__all__ = ["EdgeIndexBatch"]


@dataclass(eq=False)
class EdgeIndexBatch(BatchBase):
    """Event-partitioned graph incidence matrices.

    Node indexes are stored in a global batched namespace. Per-event node
    ``spans`` determine the offsets removed by :meth:`event`, ``split`` and
    event indexing.

    Attributes
    ----------
    spans : Union[np.ndarray, torch.Tensor]
        (B) Per-entry parent spans used to build the batch offsets.
    offsets : Union[np.ndarray, torch.Tensor]
        (B) Offsets between successive indexes in the batch, computed from
        the cumulative sum of ``spans``.
    directed : bool
        Whether the edge index is directed or undirected
    """

    data: ArrayLike
    counts: ArrayLike
    edges: ArrayLike
    batch_size: int
    spans: ArrayLike
    offsets: ArrayLike
    directed: bool

    def __init__(
        self,
        data: ArrayLike,
        counts: Sequence[int] | ArrayLike,
        spans: Sequence[int] | ArrayLike,
        directed: bool,
    ) -> None:
        """Initialize a directed or reciprocal-edge graph batch.

        If the edge index corresponds to an undirected graph, each edge
        should have its reciprocal edge immediately after, e.g.

        .. code-block:: python

            [[0,1,0,2,0,3,...],
             [1,0,2,0,3,0,...]]

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            (2, E) Batched edge index
        counts : Union[List[int], np.ndarray, torch.Tensor]
            (B) Number of index elements per entry in the batch
        spans : Union[List[int], np.ndarray, torch.Tensor]
            (B) Per-entry parent spans used to derive ``offsets``.
        directed : bool
            Whether the edge index is directed or undirected
        """
        # Initialize the base class
        super().__init__(data)

        # Normalize graph metadata to the underlying array backend
        counts = self._as_long(counts)
        spans = self._as_long(spans)

        # Do a couple of basic sanity checks
        if self._sum(counts) != data.shape[1]:
            raise ValueError("The `counts` provided do not add up to the index length")
        if len(counts) != len(spans):
            raise ValueError(
                "The number of `spans` does not match the number of `counts`"
            )
        if not directed and data.shape[1] % 2 != 0:
            raise ValueError(
                "If the edge index is undirected, it should have an "
                "even number of edge"
            )

        # Convert per-event node spans into global node offsets
        offsets = self._zeros(len(spans), None if self.is_numpy else spans.device)
        offsets[1:] = self._cumsum(spans)[:-1]

        # Get the boundaries between successive index using the counts
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.spans = spans
        self.edges = edges
        self.offsets = offsets
        self.directed = directed
        self.batch_size = len(counts)

    def __getitem__(self, batch_id: int) -> ArrayLike:
        """Returns a subset of the index corresponding to one entry.

        Parameters
        ----------
        batch_id : int
            Entry index
        """
        # Make sure the batch_id is sensible
        if batch_id >= self.batch_size:
            raise IndexError(
                f"Index {batch_id} out of bound for a batch size "
                f"of ({self.batch_size})"
            )

        # Slice the globally indexed event and restore its local node namespace
        lower, upper = self.edges[batch_id], self.edges[batch_id + 1]
        index = self.data[:, lower:upper] - self.offsets[batch_id]

        # Event indexing retains the historical row-oriented representation
        return self._transpose(index)

    def event(self, batch_id: int) -> EdgeIndexData:
        """Return one event while preserving graph and span metadata.

        Parameters
        ----------
        batch_id : int
            Event position in the batch.

        Returns
        -------
        EdgeIndexData
            Event-local ``(2, E)`` incidence matrix.
        """
        # Recover the event-local incidence matrix and scalar parent span
        index = self[batch_id]
        span = self.spans[batch_id]

        return EdgeIndexData(
            self._transpose(index),
            int(span.item() if hasattr(span, "item") else span),
            self.directed,
        )

    @property
    def index(self) -> ArrayLike:
        """Alias for the underlying data stored.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (2, E) Underlying batch of edge indexes
        """
        return self.data

    @property
    def index_t(self) -> ArrayLike:
        """Alias for the underlying data stored, transposed

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (E, 2) Underlying batch of edge indexes, transposed
        """
        return self._transpose(self.data)

    @property
    def batch_ids(self) -> ArrayLike:
        """Returns the batch ID of each element in the full index list.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Complete batch ID array, one per element
        """
        return self._repeat(self._arange(self.batch_size), self.counts)

    @property
    def directed_index(self) -> ArrayLike:
        """Index of the directed graph. If a graph is undirected, it only
        returns one of the two edges corresponding to a connection.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (2, E//2) Underlying batch of edge indexes
        """
        # If the graph is directed, nothing to do
        if self.directed:
            return self.data

        # Otherwise, skip every second edge in the index
        return self.data[:, ::2]

    @property
    def directed_index_t(self) -> ArrayLike:
        """Index of the directed graph, transposed. If the graph is undirected,
        it only returns one of the two edges corresponding to a connection.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (E//2, 2) Underlying batch of edge indexes, transposed
        """
        return self._transpose(self.directed_index)

    @property
    def directed_counts(self) -> ArrayLike:
        """Returns the number of edges per entry, counting edges once even
        if they are bidirectional.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (B) Complete batch ID array, one per element
        """
        # If the graph is directed, the counts are exact
        if self.directed:
            return self.counts

        # Otherwise, indexes are twice as long
        return self.counts // 2

    @property
    def directed_batch_ids(self) -> ArrayLike:
        """Returns the batch ID of each element in the directed index.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Complete batch ID array, one per element
        """
        return self._repeat(self._arange(self.batch_size), self.directed_counts)

    def split(self) -> list[ArrayLike]:
        """Breaks up the index batch into its constituents.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one index per entry in the batch
        """
        # Split row-oriented edges before restoring event-local node indexes
        indexes = list(self._split(self._transpose(self.index), self.splits))
        for batch_id in range(self.batch_size):
            indexes[batch_id] = indexes[batch_id] - self.offsets[batch_id]

        return indexes

    def to_numpy(self) -> "EdgeIndexBatch":
        """Cast underlying index to a `np.ndarray` and return a new instance.

        Returns
        -------
        EdgeIndexBatch
            NumPy-backed edge-index batch.
        """
        # If the underlying data is of the right type, nothing to do
        if self.is_numpy:
            return self

        data = self._to_numpy(self.data)
        counts = self._to_numpy(self.counts)
        spans = self._to_numpy(self.spans)

        return EdgeIndexBatch(data, counts, spans, self.directed)

    def to_tensor(self, dtype: Any = None, device: Any = None) -> "EdgeIndexBatch":
        """Cast underlying index to a `torch.tensor` and return a new instance.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Data type of the tensor to create
        device : torch.device, optional
            Device on which to put the tensor

        Returns
        -------
        EdgeIndexBatch
            PyTorch-backed edge-index batch.
        """
        # If the underlying data is of the right type, nothing to do
        if not self.is_numpy:
            return self

        data = self._to_tensor(self.data, dtype, device)
        counts = self._to_tensor(self.counts, dtype, device)
        spans = self._to_tensor(self.spans, dtype, device)

        return EdgeIndexBatch(data, counts, spans, self.directed)
