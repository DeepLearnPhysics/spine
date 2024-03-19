"""Module with data classes of objects which represent batches of data.

Two batched data structures exist:
- `TensorBatch`: All-purpose structure of np.ndarray/torch.Tensor data
- `IndexBatch`: Specifically geared at arranging indexes which point at
  specific sections of a TensorBatch object.
"""

import numpy as np
import torch
from dataclasses import dataclass
from warnings import warn
from typing import Union, List

from mlreco.utils.globals import BATCH_COL, COORD_COLS
from mlreco.utils.torch_local import unique_index
from mlreco.utils.decorators import inherit_docstring

from .meta import Meta


@dataclass
class BatchBase:
    """Base class for all types of batched data.

    Attributes
    ----------
    data : Union[np.ndarray, torch.Tensor]
        Batched data
    counts : Union[np.ndarray, torch.Tensor]
        (B) Number of data elements in each entry of the batch
    edges : Union[np.ndarray, torch.Tensor]
        (B+1) Edges separating the entries in the batch
    batch_size : int
        Number of entries that make up the batched data
    """
    data: Union[np.ndarray, torch.Tensor]
    counts: Union[np.ndarray, torch.Tensor]
    edges: Union[np.ndarray, torch.Tensor]
    batch_size: int

    def __init__(self, data, is_sparse=False, is_list=False):
        """Shared initializations across all types of batched data.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            Batched data
        is_sparse : bool, default False
            If initializing from an ME sparse data, flip to True
        is_list : bool, default False
            Whether the underlying data is a list of tensors
        """
        # Store the datatype
        self.is_numpy = not is_sparse and not isinstance(data, torch.Tensor)
        self.is_sparse = is_sparse
        self.is_list = is_list

        # Store the datatype
        self.dtype = data.dtype

        # Store the device
        self.device = None
        if not self.is_numpy:
            ref = data if not is_sparse else data.F
            self.device = ref.device

    def __len__(self):
        """Returns the number of entries that make up the batch."""
        return self.batch_size

    @property
    def shape(self):
        """Shape of the underlying data.

        Returns
        -------
        tuple
            Tuple of sizes in each dimension
        """
        if not self.is_list:
            return self.data.shape
        else:
            return len(self.data)

    @property
    def splits(self):
        """Boundaries needed to split the data into its constituents.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (B-1) One split per batch boundary
        """
        return self.edges[1:-1]

    def get_counts(self, batch_ids, batch_size):
        """Finds the number of elements in each entry, provided a batch ID list.

        Parameters
        ----------
        batch_ids : Union[np.ndarray, torch.Tensor]
            List of batch IDs
        batch_size : int
            Number of entries that make up the batched data

        Returns
        -------
        np.ndarray
            (B) Length of each entry
        """
        # Get the count list
        device = None if self.is_numpy else batch_ids.device
        counts = self._zeros(batch_size, device)
        if len(batch_ids):
            # Find the length of each batch ID in the input index
            uni, cnts = self._unique(batch_ids)
            counts[self._as_long(uni)] = cnts

        return counts

    def get_edges(self, counts):
        """Finds the edges between successive entries in the batch.

        Parameters
        ----------
        counts : Union[np.ndarray, torch.Tensor]
            (B)Length of each entry

        Returns
        -------
        np.ndarray
            (B+1) Edges of successive entries in the batch
        """
        # Get the edge list
        device = None if self.is_numpy else counts.device
        edges = self._zeros(len(counts)+1, device)
        cumsum = self._cumsum(counts)
        edges[1:] = cumsum

        return edges

    def _empty(self, x):
        if self.is_numpy:
            return np.empty(x, dtype=np.int64)
        else:
            return torch.empty(x, dtype=torch.long, device=self.device)

    def _zeros(self, x, device=None):
        if self.is_numpy:
            return np.zeros(x, dtype=np.int64)
        else:
            return torch.zeros(x, dtype=torch.long, device=device)

    def _ones(self, x):
        if self.is_numpy:
            return np.ones(x, dtype=np.int64)
        else:
            return torch.ones(x, dtype=torch.long, device=self.device)

    def _as_long(self, x):
        if self.is_numpy:
            return np.asarray(x, dtype=np.int64)
        else:
            # Always on CPU. This is because splits are supposed to be on
            # CPU regardless of the location of the underlying data
            return torch.as_tensor(x, dtype=torch.long, device='cpu')

    def _unique(self, x):
        if self.is_numpy:
            return np.unique(x, return_counts=True)
        else:
            return torch.unique(x, return_counts=True)

    def _transpose(self, x):
        if self.is_numpy:
            return np.transpose(x)
        else:
            return torch.transpose(x, 0, 1)

    def _sum(self, x):
        if self.is_numpy:
            return np.sum(x)
        else:
            return torch.sum(x)

    def _cumsum(self, x):
        if self.is_numpy:
            return np.cumsum(x)
        else:
            return torch.cumsum(x, dim=0)

    def _arange(self, x):
        if self.is_numpy:
            return np.arange(x)
        else:
            return torch.arange(x, device=self.device)

    def _cat(self, x):
        if self.is_numpy:
            return np.conctenate(x)
        else:
            return torch.cat(x, dim=0)

    def _split(self, *x):
        if self.is_list:
            return np.split(*x)
        else:
            return np.split(*x) if self.is_numpy else torch.tensor_split(*x)

    def _stack(self, x):
        return np.vstack(x) if self.is_numpy else torch.stack(x)

    def _repeat(self, *x):
        return np.repeat(*x) if self.is_numpy else torch.repeat(*x)


@dataclass
@inherit_docstring(BatchBase)
class TensorBatch(BatchBase):
    """Batched tensor with the necessary methods to slice it."""

    def __init__(self, data, counts=None, batch_size=None, is_sparse=False):
        """Initialize the attributes of the class.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor, ME.SparseTensor]
            (N, C) Batched data where the batch column is `BATCH_COL`
        counts : Union[List, np.ndarray, torch.Tensor]
            (B) Number of data rows in each entry
        batch_size : int, optional
            Number of entries that make up the batched data
        is_sparse : bool, default False
            If initializing from an ME sparse data, flip to True
        """
        # Initialize the base class
        super().__init__(data, is_sparse=is_sparse)

        # Should provide either the counts, or the batch size
        assert (counts is not None) ^ (batch_size is not None), (
                "Provide either `counts` or `batch_size`, not both")

        # If the number of batches is not provided, get it from the counts
        if batch_size is None:
            batch_size = len(counts)

        # If the counts are not provided, must build them once
        if counts is None:
            # Define the array functions depending on the input type
            ref = data if not is_sparse else data.C
            counts = self.get_counts(ref[:, BATCH_COL], batch_size)

        # Cast
        counts = self._as_long(counts)
        assert self._sum(counts) == len(data), (
                "The `counts` provided do not add up to the tensor length")

        # Get the boundaries between entries in the batch
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.edges = edges
        self.batch_size = batch_size

    def __getitem__(self, batch_id):
        """Returns a subset of the tensor corresponding to one entry.

        Parameters
        ----------
        batch_id : int
            Entry index
        """
        # Make sure the batch_id is sensible
        if batch_id >= self.batch_size:
            raise IndexError(f"Index {batch_id} out of bound for a batch size "
                             f"of ({self.batch_size})")

        # Return
        lower, upper = self.edges[batch_id], self.edges[batch_id + 1]
        if not self.is_sparse:
            return self.data[lower:upper]
        else:
            from MinkowskiEngine import SparseTensor
            return SparseTensor(
                    self.data.F[lower:upper],
                    coordinates=self.data.C[lower:upper])

    @property
    def tensor(self):
        """Alias for the underlying data stored.

        Returns
        -------
        Union[np.ndarray, torch.Tensor, ME.SparseTensor]
            Underlying tensor of data
        """
        return self.data

    def split(self):
        """Breaks up the tensor batch into its constituents.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one tensor per entry in the batch
        """
        if not self.is_sparse:
            return self._split(self.data, self.splits)
        else:
            from MinkowskiEngine import SparseTensor
            coords = self._split(self.data.C, self.splits)
            feats = self._split(self.data.F, self.splits)
            return [SparseTensor(
                feats[i], coordinates=coords[i]) for i in self.batch_size]

    def to_numpy(self):
        """Cast underlying tensor to a `np.ndarray` and return a new instance.

        Returns
        -------
        TensorBatch
            New `TensorBatch` object with an underlying np.ndarray tensor.
        """
        # If the underlying data is of the right type, nothing to do
        if self.is_numpy:
            return self

        data = self.data
        if self.is_sparse:
            data = torch.cat([self.data.C.float(), self.data.F], dim=1)

        to_numpy = lambda x: x.cpu().detach().numpy()
        data = to_numpy(data)
        counts = to_numpy(self.counts)

        return TensorBatch(data, counts)

    def to_tensor(self, dtype=None, device=None):
        """Cast underlying tensor to a `torch.tensor` and return a new instance.

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

        to_tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        data = to_tensor(self.data)
        counts = to_tensor(self.counts)

        return TensorBatch(data, counts)

    def to_cm(self, meta):
        """Converts the coordinates of the tensor to cm.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.is_numpy, "Can only convert units of numpy arrays"
        self.data[:, COORD_COLS] = meta.to_cm(self.data[:, COORD_COLS])

    def to_pixel(self, meta):
        """Converts the coordinates of the tensor to pixel indexes.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.is_numpy, "Can only convert units of numpy arrays"
        self.data[:, COORD_COLS] = meta.to_pixel(self.data[:, COORD_COLS])

    @classmethod
    def from_list(cls, data_list):
        """Builds a batch from a list of tensors.

        Parameters
        ----------
        data_list : List[Union[np.ndarray, torch.Tensor]]
            List of tensors, exactly one per batch
        """
        # Check that we are not fed an empty list of tensors
        assert len(data_list), (
                "Must provide at least one tensor to build a tensor batch")
        is_numpy = not isinstance(data_list[0], torch.Tensor)

        # Compute the counts from the input list
        counts = [len(t) for t in data_list]

        # Concatenate input
        if is_numpy:
            return cls(np.concatenate(data_list, axis=0), counts)
        else:
            return cls(torch.cat(data_list, dim=0), counts)


@dataclass
@inherit_docstring(BatchBase)
class IndexBatch(BatchBase):
    """Batched index with the necessary methods to slice it.

    Attributes
    ----------
    offsets : Union[List, np.ndarray, torch.Tensor]
        (B) Offsets between successive indexes in the batch
    full_counts : Union[List, np.ndarray, torch.Tensor]
        (B) Number of index elements per entry in the batch. This
        is the same as counts if the underlying data is a single index
    """
    offsets: Union[np.ndarray, torch.Tensor]
    full_counts: Union[np.ndarray, torch.Tensor]

    def __init__(self, data, offsets, counts=None, full_counts=None,
                 batch_ids=None, batch_size=None, is_numpy=True):
        """Initialize the attributes of the class.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor, 
                     List[Union[np.ndarray, torch.Tensor]]]
            Simple batched index or list of indexes
        offsets : Union[List, np.ndarray, torch.Tensor]
            (B) Offsets between successive indexes in the batch
        counts : Union[List, np.ndarray, torch.Tensor], optional
            (B) Number of indexes in the batch
        full_counts : Union[List, np.ndarray, torch.Tensor], optional
            (B) Number of index elements per entry in the batch. This
            is the same as counts if the underlying data is a single index
        batch_ids : Union[List, np.ndarray, torch.Tensor], optional
            (I) Batch index of each of the clusters. If not specified, the
            assumption is that each count corresponds to a specific entry
        batch_size : int, optional
            Number of entries in the batch. Must be specified along batch_ids
        is_numpy : bool, default True
            Weather the underlying representation is `np.ndarray` or
            `torch.Tensor`. Must specify if the input list is empty
        """
        # Check weather the input is a single index or a list
        is_list = isinstance(data, (list, tuple)) or data.dtype == object

        # Initialize the base class
        if not is_list:
            init_data = data
        elif len(data):
            init_data = data[0]
        else:
            warn("The input list is empty, underlying data type arbitrary.")
            init_data = np.empty(0, dtype=np.int64)

        super().__init__(init_data, is_list=is_list)

        # Get the counts if they are not provided for free
        if counts is None:
            assert batch_ids is not None and batch_size is not None, (
                    "Must provide `batch_size` alongside `batch_ids`.")
            counts = self.get_counts(batch_ids, batch_size)

        else:
            batch_size = len(counts)

        # Get the number of index elements per entry in the batch
        if full_counts is None:
            assert not self.is_list, (
                    "When initializing an index list, provide `full_counts`")
            full_counts = counts

        # Cast
        counts = self._as_long(counts)
        full_counts = self._as_long(full_counts)
        offsets = self._as_long(offsets)

        # Do a couple of basic sanity checks
        assert self._sum(counts) == len(data), (
                "The `counts` provided do not add up to the index length")
        assert len(counts) == len(offsets), (
                "The number of `offsets` does not match the number of `counts`")

        # Get the boundaries between successive index using the counts
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.full_counts = full_counts
        self.edges = edges
        self.offsets = offsets
        self.batch_size = batch_size

    def __len__(self):
        """Returns the number of entries that make up the batch."""
        return self.batch_size

    def __getitem__(self, batch_id):
        """Returns a subset of the index corresponding to one entry.

        Parameters
        ----------
        batch_id : int
            Entry index
        """
        # Make sure the batch_id is sensible
        if batch_id >= self.batch_size:
            raise IndexError(f"Index {batch_id} out of bound for a batch size "
                             f"of ({self.batch_size})")

        # Return
        lower, upper = self.edges[batch_id], self.edges[batch_id + 1]
        if not self.is_list:
            return self.data[lower:upper] - self.offsets[batch_id]
        else:
            entry = np.empty(upper-lower, dtype=object)
            entry[:] = self.data[lower:upper]
            return entry - self.offsets[batch_id]

    @property
    def index(self):
        """Alias for the underlying data stored.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Underlying index
        """
        assert not self.is_list, (
                "Underlying data is not a single index, use `index_list`")

        return self.data

    @property
    def index_list(self):
        """Alias for the underlying data list stored.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            Underlying index list
        """
        assert self.is_list, (
                "Underlying data is a single index, use `index`")

        return self.data

    @property
    def full_index(self):
        """Returns the index combining all sub-indexes, if relevant.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Complete concatenated index
        """
        if not self.is_list:
            return self.data
        else:
            return self._cat(self.data) if len(self.data) else self._empty(0)

    @property
    def batch_ids(self):
        """Returns the batch ID of each index in the list.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (I) Batch ID array, one per index in the list
        """
        return self._repeat(self._arange(self.batch_size), self.counts)

    @property
    def full_batch_ids(self):
        """Returns the batch ID of each element in the full index list.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Complete batch ID array, one per element
        """
        return self._repeat(self._arange(self.batch_size), self.full_counts)

    def split(self):
        """Breaks up the index batch into its constituents.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one index per entry in the batch
        """
        indexes = self._split(self.data, self.splits)
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

        to_numpy = lambda x: x.cpu().detach().numpy()
        if not self.is_list:
            data = to_numpy(self.data)
        else:
            data = np.empty(len(data), dtype=object)
            for i in range(len(self.data)):
                data[i] = to_numpy(self.data[i])

        offsets = to_numpy(self.offsets)
        counts = to_numpy(self.counts)
        full_counts = to_numpy(self.full_counts)

        return IndexBatch(data, offsets, counts, full_counts)

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

        to_tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        if not self.is_list:
            data = to_tensor(self.data)
        else:
            data = np.empty(len(data), dtype=object)
            for i in range(len(self.data)):
                data[i] = to_tensor(self.data[i])

        offsets = to_tensor(self.offsets)
        counts = to_tensor(self.counts)
        full_counts = to_tensor(self.full_counts)

        return IndexBatch(index, offsets, counts, full_counts)


@dataclass
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
        counts : Union[List, np.ndarray, torch.Tensor]
            (B) Number of index elements per entry in the batch
        offsets : Union[List, np.ndarray, torch.Tensor]
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
        assert self._sum(counts) == data.shape[1], (
                "The `counts` provided do not add up to the index length")
        assert len(counts) == len(offsets), (
                "The number of `offsets` es not match the number of `counts`")
        if not directed:
            assert data.shape[1]%2 == 0, (
                    "If the edge index is undirected, it should have an "
                    "even number of edge")

        # Get the boundaries between successive index using the counts
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.edges = edges
        self.offsets = offsets
        self.directed = directed
        self.batch_size = len(counts)

    def __len__(self):
        """Returns the number of entries that make up the batch."""
        return self.batch_size

    def __getitem__(self, batch_id):
        """Returns a subset of the index corresponding to one entry.

        Parameters
        ----------
        batch_id : int
            Entry index
        """
        # Make sure the batch_id is sensible
        if batch_id >= self.batch_size:
            raise IndexError(f"Index {batch_id} out of bound for a batch size "
                             f"of ({self.batch_size})")

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
        return self.data[:,::2]

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
        return self.counts//2

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
            indexes[batch_id] = indexes[batch_id] - self._offsets[batch_id]

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

        to_numpy = lambda x: x.cpu().detach().numpy()
        data = to_numpy(self.data)
        counts = to_numpy(self.counts)
        offsets = to_numpy(self.offsets)

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

        to_tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        data = to_tensor(self.data)
        counts = to_tensor(self.counts)
        offsets = to_tensor(self.offsets)

        return EdgeIndexBatch(data, counts, offsets, self.directed)
