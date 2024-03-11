"""Module with data classes of objects which represent batches of data.

Two batched data structures exist:
- `TensorBatch`: All-purpose structure of np.ndarray/torch.Tensor data
- `IndexBatch`: Specifically geared at arranging indexes which point at
  specific sections of a TensorBatch object.
"""

import numpy as np
import torch
from dataclasses import dataclass
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
    counts : Union[List, np.ndarray, torch.Tensor]
        (B) Number of data elements in each entry of the batch
    edges : Union[List, np.ndarray, torch.Tensor]
        (B+1) Edges separating the entries in the batch
    batch_size : int
        Number of entries that make up the batched data
    """
    data: Union[np.ndarray, torch.Tensor]
    counts: Union[np.ndarray, torch.Tensor]
    edges: Union[np.ndarray, torch.Tensor]
    batch_size: int

    def __init__(self, is_numpy, is_parse):
        """Shared initializations across all types of batched data.

        Parameters
        ----------
        is_numpy : Whether the data is a `np.ndarray` or not
            Batched data
        is_sparse : bool, default False
            If initializing from an ME sparse data, flip to True
        """
        # Store the datatype
        self.is_numpy = is_numpy
        self.is_sparse = is_sparse

        # Fetch datatype-specific functions once and for all
        if self.is_numpy:
            self._empty = lambda x: np.zeros(x, dtype=np.int64)
            self._zeros = lambda x: np.zeros(x, dtype=np.int64)
            self._ones = lambda x: np.zeros(x, dtype=np.int64)
            self._aslong = lambda x: np.zeros(x, dtype=np.int64)
            self._unique = lambda x: np.unique(x, return_counts=True)
            self._cumsum = np.cumsum
            self._arange = np.arange
            self._cat = np.concatenate
            self._stack = np.vstack
            self._split = np.split
            self._repeat = np.repeat
        else:
            dlong, device = torch.long, self.data.device
            self._empty = lambda x: torch.empty(x, dtype=dlong, device=device)
            self._zeros = lambda x: torch.zeros(x, dtype=dlong, device=device)
            self._ones = lambda x: torch.ones(x, dtype=dlong, device=device)
            self._aslong = lambda x: x.long()
            self._unique = lambda x: torch.unique(x, return_counts=True)
            self._cumsum = lambda x: torch.cumsum(x, dim=0)
            self._arange = lambda x: torch.arange(x, device=index.device)
            self._cat = lambda x: torch.cat(x, dim=0)
            self._stack = torch.cat
            self._split = torch.tensor_split
            self._repeat = torch.repeat_interleave

    def __len__(self):
        """Returns the number of entries that make up the batch."""
        return self.batch_size

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
        counts = self._zeros(self.batch_size)
        if len(batch_ids):
            # Find the length of each batch ID in the input index
            uni, cnts = self._unique(batch_ids, batch_size)
            counts = self._zeros(self.batch_size)
            counts[self._aslong(uni)] = cnts

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
        edges = self._empty(len(counts)+1)
        cumsum = self._cumsum(counts)
        edges[1:] = cumsum

        return edges


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
        is_numpy = not is_sparse and not isinstance(data, torch.Tensor)
        super().__init__(is_numpy, is_sparse)

        # Should provide either the counts, or the batch size
        assert (counts is not None) ^ (batch_size is not None), (
                "Provide either `counts` or `batch_size`, not both")

        # If the number of batches is not provided, get it from the counts
        if batch_size is None:
            batch_size = len(counts)

        # If the counts are not provided, must build them once
        if counts is None:
            # Define the array functions depending on the input type
            ref = data if not sparse else data.C
            counts = self.get_counts(ref[:, BATCH_COL], batch_size)

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
            feat = self._split(self.data.F, self.splits)
            return [SparseTensor(
                feats[i], coordinates=coords[i]) for i in self.batch_size]

    def to_numpy(self):
        """Cast underlying tensor to a `np.ndarray` and return a new instance.

        Returns
        -------
        TensorBatch
            New `TensorBatch` object with an underlying np.ndarray tensor.
        """
        assert not self.is_numpy, (
                "Must be a `torch.Tensor` to be cast to `np.ndarray`")

        data = self.data
        if self.is_sparse:
            data = torch.cat([self.data.C.float(), self.data.F], dim=1)

        data = data.cpu().detach().numpy()
        counts = self.counts.cpu().detach().numpy()

        return TensorBatch(data, counts)

    def to_data(self, dtype=None, device=None):
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
        assert self.is_numpy, (
                "Must be a `np.ndarray` to be cast to `torch.Tensor`")

        data = torch.as_tensor(self.data, dtype=dtype, device=device)
        counts = torch.as_tensor(self.counts, dtype=torch.int64, device=device)

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
    """
    offsets: Union[np.ndarray, torch.Tensor]

    def __init__(self, data, counts, offsets, batch_ids=None,
                 batch_size=None, is_numpy=True):
        """Initialize the attributes of the class.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor, 
                     List[Union[np.ndarray, torch.Tensor]]]
            Simple batched index or list of indexes
        counts : Union[List, np.ndarray, torch.Tensor]
            (B) Number of indexes in the batch
        offsets : Union[List, np.ndarray, torch.Tensor]
            (B) Offsets between successive indexes in the batch
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
        self.is_list = isinstance(data, (list, tuple)) or data.dtype == object

        # Initialize the base class
        if self.is_list and len(data):
            is_numpy = not isinstance(data[0], torch.Tensor)
        elif not self.is_list
            is_numpy = not isinstance(data, torch.Tensor)
        super().__init__(is_numpy)

        # Get the boundaries between successive index using the counts
        edges = self.get_edges(counts)

        # If batch_ids and batch_size are specified, build a count list
        if counts is None:
            if batch_ids is not None or batch_size is not None:
                assert batch_ids is not None and batch_size is not None, (
                        "Must provide `batch_size` alongside `batch_ids`.")

                list_counts = get_counts(batch_ids, batch_size, self.is_numpy)
                list_edges = get_edges(list_counts, self.is_numpy)

            else:
                batch_size = len(counts)
                batch_ids = self._arange(batch_size)
                list_edges = None

        else:
            assert batch_ids is None and batch_size is None, (
                    "Should not provide both `list_counts` along with either "
                    "`batch_ids` or `batch_size`")

            list_edges = self.get_edges(list_counts)
            batch_size = len(list_counts)
            batch_ids = self._repeat(self._arange(batch_size), list_counts)

        # Compute the absolute boundaries between successive entries
        if list_counts is None:
            assert len(counts) == len(offsets), (
                    "There should be as many offsets and counts as "
                    "there are entries in the batch.")

            batch_counts = counts
            batch_edges = edges
        else:
            assert len(list_counts) == len(offsets), (
                    "There should be as many offsets and list_counts as "
                    "there are entries in the batch.")

            batch_counts = self._empty(batch_size)
            for batch_id in range(batch_size):
                lower, upper = list_edges[batch_id], list_edges[batch_id + 1]
                batch_counts[batch_id] = edges[upper] - edges[lower]

            batch_edges = self.get_edges(batch_counts)

        # Store the attributes
        self.data = data
        self.data_list = data_list
        self.counts = counts
        self.edges = edges
        self.offsets = offsets
        self.batch_counts = batch_counts
        self.batch_edges = batch_edges
        self.list_counts = list_counts
        self.list_edges = list_edges
        self.batch_size = batch_size
        self.batch_ids = batch_ids

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
        if self.list_edges is None:
            lower, upper = self.edges[batch_id], self.edges[batch_id + 1]

            return self.index[lower:upper] - self.offsets[batch_id]

        else:
            lower = self.list_edges[batch_id]
            upper = self.list_edges[batch_id + 1]

            return self.index_list[lower:upper] - self.offsets[batch_id]

    @property
    def index(self):
        """Alias for the underlying data stored.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Underlying tensor of data
        """
        return self.data

    @property
    def index_list(self):
        """Alias for the underlying data list stored.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Underlying tensor of data
        """
        return self.data_list

    @property
    def list_splits(self):
        """Boundaries needed to split the input into one list per entry.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (B-1) One split per batch boundary
        """
        if self.list_edges is None:
            return None

        return self.list_edges[1:-1]

    @property
    def full_batch_ids(self):
        """Returns the batch ID of each element in the full index list.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Complete batch ID array, one per element
        """
        return self._repeat(self.batch_ids, self.counts)

    def split(self):
        """Breaks up the index batch into its constituents.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one index per entry in the batch
        """
        if self.list_edges is None:
            indexes = self._split(self.index, self.splits)
            for batch_id in range(self.batch_size):
                indexes[batch_id] = indexes[batch_id] - self.offsets[batch_id]
        else:
            indexes = np.split(self.index_list, self.list_splits)
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
        assert not self.is_numpy, (
                "Must be a `torch.Tensor` to be cast to `np.ndarray`")

        index = self.index.cpu().detach().numpy()
        counts = self.counts.cpu().detach().numpy()
        offsets = self.offsets.cpu().detach().numpy()
        list_counts = None
        if self.list_counts is not None:
            list_counts = self.list_counts.cpu().detach().numpy()

        return IndexBatch(index, counts, offsets, list_counts)

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
        assert self.is_numpy, (
                "Must be a `np.ndarray` to be cast to `torch.Tensor`")

        index = torch.as_tensor(self.index, dtype=dtype, device=device)
        counts = torch.as_tensor(self.counts, dtype=torch.long, device=device)
        offsets = torch.as_tensor(self.offsets, dtype=torch.long, device=device)
        list_counts = None
        if self.list_counts is not None:
            list_counts = torch.as_tensor(
                    self.list_counts, dtype=torch.long, device=device)

        return IndexBatch(index, counts, offsets, list_counts)


@dataclass
class EdgeIndexBatch:
    """Batched edge index with the necessary methods to slice it.

    Attributes
    ----------
    index : Union[np.ndarray, torch.Tensor]
        (2, E) Batched edge index
    counts : Union[List, np.ndarray, torch.Tensor]
        (B) Number of index elements per entry in the batch
    edges : Union[List, np.ndarray, torch.Tensor]
        (B+1) Edges of the indexes in the batch
    offsets : Union[List, np.ndarray, torch.Tensor]
        (B) Offsets between successive indexes in the batch
    directed : bool
        Whether the edge index is directed or undirected
    batch_size : int
        Number of entries that make up the batched tensor
    """
    index: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    offsets: np.ndarray
    directed: bool
    batch_size: np.ndarray

    def __init__(self, index, counts, offsets, directed):
        """Initialize the attributes of the class.

        Parameters
        ----------
        index : Union[np.ndarray, torch.Tensor]
            (E) Batched edge index
        counts : Union[List, np.ndarray, torch.Tensor]
            (B) Number of index elements per entry in the batch
        offsets : Union[List, np.ndarray, torch.Tensor]
            (B) Offsets between successive indexes in the batch
        directed : bool
            Whether the edge index is directed or undirected
        """
        # Check the typing of the input
        self.is_numpy = not isinstance(index, torch.Tensor)

        # Get the boundaries between successive index using the counts
        edges = get_edges(counts, self.is_numpy)

        # Store the attributes
        self.index = index
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
        index = self.index[lower:upper] - self.offsets[batch_id]
        if directed:
            return index
        else:
            return self._stack(index, index[:, ::-1])

    @property
    def splits(self):
        """Boundaries needed to split the input into its constituents.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (I-1) One split per index boundary
        """
        return self.edges[1:-1]

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
    def full_index(self):
        """For undirectly graph, adds reciprocal edges to the underlying index.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Complete batch ID array, one per element
        """
        # If the graph is directed, nothing to do
        if self.directed:
            return self.index

        # Otherwise, add reciprocal edges to each entry in the batch
        full_index = self._empty((2*self.index.shape[0], self.index.shape[1]))
        for batch_id in range(self.batch_size):
            offset = 0 if batch_id < 1 else 2*self.counts[batch_id - 1]
            mask = offset + self._arange(self.counts[batch_id])
            lower, upper = self.edges[batch_id], self.edges[batch_id + 1]
            full_index[mask] = self.index[lower:upper]
            full_index[len(mask) + mask] = self.index[lower:upper][:,::-1]

        return full_index

    def split(self):
        """Breaks up the index batch into its constituents.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one index per entry in the batch
        """
        indexes = self._split(self.index, self.splits)
        for batch_id in range(self.batch_size):
            if not directed:
                indexes[i] = self._stack(indexes[i], indexes[:, ::-1])
            indexes[i] = indexes[i] - self._offsets[batch_id]

        return indexes

    def to_numpy(self):
        """Cast underlying index to a `np.ndarray` and return a new instance.

        Returns
        -------
        TensorBatch
            New `TensorBatch` object with an underlying np.ndarray tensor.
        """
        assert not self.is_numpy, (
                "Must be a `torch.Tensor` to be cast to `np.ndarray`")

        index = self.index.cpu().detach().numpy()
        counts = self.counts.cpu().detach().numpy()
        offsets = self.offsets.cpu().detach().numpy()

        return EdgeIndexBatch(index, counts, offsets, self.directed)

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
        assert self.is_numpy, (
                "Must be a `np.ndarray` to be cast to `torch.Tensor`")

        index = torch.as_tensor(self.index, dtype=dtype, device=device)
        counts = torch.as_tensor(self.counts, dtype=torch.long, device=device)
        offsets = torch.as_tensor(self.offsets, dtype=torch.long, device=device)

        return EdgeIndexBatch(index, counts, offsets, self.directed)


def get_counts(batch_ids, batch_size, is_numpy):
    """Finds the length of each entry, provided a batch ID list.

    Parameters
    ----------
    batch_ids : np.ndarrary
        List of batch IDs
    batch_size : int, optional
        Number of entries that make up the batched index
    is_numpy : bool
        Whether the input is a number array or not

    Returns
    -------
    np.ndarray
        (B) Length of each entry
    """
    # Define the array functions depending on the input type
    if is_numpy:
        zeros = lambda x: np.zeros(x, dtype=np.int64)
        unique = lambda x: np.unique(x, return_counts=True)
        cast = lambda x: x.astype(np.int64)
    else:
        zeros = lambda x: torch.zeros(
                x, dtype=torch.long, device=batch_ids.device)
        unique = lambda x: torch.unique(x, return_counts=True)
        cast = lambda x: x.long()

    # Get the count list
    counts = zeros(batch_size)
    if len(batch_ids):
        # Find the length of each batch ID in the input index
        uni, cnts = unique(batch_ids)
        counts = zeros(batch_size)
        counts[cast(uni)] = cnts

    return counts


def get_edges(counts, is_numpy):
    """Finds the edges between successive entries in the batch.

    Parameters
    ----------
    counts : Union[np.ndarray, torch.Tensor]
        (B)Length of each entry
    is_numpy : bool
        Whether the input is a number array or not

    Returns
    -------
    np.ndarray
        (B+1) Edges of successive entries in the batch
    """
    # Define the array functions depending on the input type
    if is_numpy:
        empty = lambda x: np.zeros(x, dtype=np.int64)
        cumsum = np.cumsum
    else:
        empty = lambda x: torch.zeros(
                x, dtype=torch.long, device=counts.device)
        cumsum = lambda x: torch.cumsum(x, dim=0)

    # Get the edge list
    edges = empty(len(counts)+1)
    cumsum = cumsum(counts)
    edges[1:] = cumsum

    return edges
