"""Module with a dataclass targeted at a batched edge index.

An edge index is a sparse representation of a graph incidence matrix.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np

from spine.utils.conditional import torch
from spine.utils.docstring import inherit_docstring

from .base import BatchBase

__all__ = ["EdgeIndexBatch"]


@dataclass(eq=False)
@inherit_docstring(BatchBase)
class EdgeIndexBatch(BatchBase):
    """Batched edge index with the necessary methods to slice it.

    Attributes
    ----------
    offsets : Union[np.ndarray, torch.Tensor]
        (B) Offsets between successive indexes in the batch
    directed : bool
        Whether the edge index is directed or undirected
    """

    offsets: Union[np.ndarray, torch.Tensor]
    directed: bool

    def __init__(self, data, counts, offsets, directed):
        """Initialize the attributes of the class.

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
        offsets : Union[List[int], np.ndarray, torch.Tensor]
            (B) Offsets between successive indexes in the batch
        directed : bool
            Whether the edge index is directed or undirected
        """
        # Initialize the base class
        super().__init__(data)

        # Cast
        counts = self._as_long(counts)
        offsets = self._as_long(offsets)

        # Do a couple of basic sanity checks
        assert (
            self._sum(counts) == data.shape[1]
        ), "The `counts` provided do not add up to the index length"
        assert len(counts) == len(
            offsets
        ), "The number of `offsets` es not match the number of `counts`"
        if not directed:
            assert data.shape[1] % 2 == 0, (
                "If the edge index is undirected, it should have an "
                "even number of edge"
            )

        # Get the boundaries between successive index using the counts
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.edges = edges
        self.offsets = offsets
        self.directed = directed
        self.batch_size = len(counts)

    def __getitem__(self, batch_id):
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

        # Return
        lower, upper = self.edges[batch_id], self.edges[batch_id + 1]
        index = self.data[:, lower:upper] - self.offsets[batch_id]
        return self._transpose(index)

    @property
    def index(self):
        """Alias for the underlying data stored.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (2, E) Underlying batch of edge indexes
        """
        return self.data

    @property
    def index_t(self):
        """Alias for the underlying data stored, transposed

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (E, 2) Underlying batch of edge indexes, transposed
        """
        return self._transpose(self.data)

    @property
    def batch_ids(self):
        """Returns the batch ID of each element in the full index list.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Complete batch ID array, one per element
        """
        return self._repeat(self._arange(self.batch_size), self.counts)

    @property
    def directed_index(self):
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
    def directed_index_t(self):
        """Index of the directed graph, transposed. If the graph is undirected,
        it only returns one of the two edges corresponding to a connection.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (E//2, 2) Underlying batch of edge indexes, transposed
        """
        return self._transpose(self.directed_index)

    @property
    def directed_counts(self):
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
    def directed_batch_ids(self):
        """Returns the batch ID of each element in the directed index.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Complete batch ID array, one per element
        """
        return self._repeat(self._arange(self.batch_size), self.directed_counts)

    def split(self):
        """Breaks up the index batch into its constituents.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one index per entry in the batch
        """
        indexes = self._split(self._transpose(self.index), self.splits)
        for batch_id in range(self.batch_size):
            indexes[batch_id] = indexes[batch_id] - self.offsets[batch_id]

        return indexes

    def to_numpy(self):
        """Cast underlying index to a `np.ndarray` and return a new instance.

        Returns
        -------
        TensorBatch
            New `TensorBatch` object with an underlying np.ndarray tensor.
        """
        # If the underlying data is of the right type, nothing to do
        if self.is_numpy:
            return self

        data = self._to_numpy(self.data)
        counts = self._to_numpy(self.counts)
        offsets = self._to_numpy(self.offsets)

        return EdgeIndexBatch(data, counts, offsets, self.directed)

    def to_tensor(self, dtype=None, device=None):
        """Cast underlying index to a `torch.tensor` and return a new instance.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Data type of the tensor to create
        device : torch.device, optional
            Device on which to put the tensor

        Returns
        -------
        TensorBatch
            New `TensorBatch` object with an underlying np.ndarray tensor.
        """
        # If the underlying data is of the right type, nothing to do
        if not self.is_numpy:
            return self

        data = self._to_tensor(self.data, dtype, device)
        counts = self._to_tensor(self.counts, dtype, device)
        offsets = self._to_tensor(self.offsets, dtype, device)

        return EdgeIndexBatch(data, counts, offsets, self.directed)
