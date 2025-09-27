"""Module with a dataclass targeted at a batch index or list of indexes."""

from dataclasses import dataclass
from typing import Union
from warnings import warn

import numpy as np

from spine.utils.conditional import torch
from spine.utils.docstring import inherit_docstring

from .base import BatchBase

__all__ = ["IndexBatch"]


@dataclass(eq=False)
@inherit_docstring(BatchBase)
class IndexBatch(BatchBase):
    """Batched index with the necessary methods to slice it.

    Attributes
    ----------
    offsets : Union[np.ndarray, torch.Tensor]
        (B) Offsets between successive indexes in the batch
    single_counts : Union[np.ndarray, torch.Tensor]
        (I) Number of index elements per index in the index list. This
        is the same as counts if the underlying data is a single index
    """

    offsets: Union[np.ndarray, torch.Tensor]
    single_counts: Union[np.ndarray, torch.Tensor]

    def __init__(
        self,
        data,
        offsets,
        counts=None,
        single_counts=None,
        batch_ids=None,
        batch_size=None,
        default=None,
        is_numpy=True,
    ):  # TODO is_numpy does nothing
        """Initialize the attributes of the class.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor,
                     List[Union[np.ndarray, torch.Tensor]]]
            Simple batched index or list of indexes
        offsets : Union[List[int], np.ndarray, torch.Tensor]
            (B) Offsets between successive indexes in the batch
        counts : Union[List[int], np.ndarray, torch.Tensor], optional
            (B) Number of indexes in the batch
        single_counts : Union[List[int], np.ndarray, torch.Tensor], optional
            (I) Number of index elements per index in the index list. This
            is the same as counts if the underlying data is a single index
        batch_ids : Union[List[int], np.ndarray, torch.Tensor], optional
            (I) Batch index of each of the clusters. If not specified, the
            assumption is that each count corresponds to a specific entry
        batch_size : int, optional
            Number of entries in the batch. Must be specified along batch_ids
        is_numpy : bool, default True
            Default type of index. Provide if `data` may be empty
        """
        # Check weather the input is a single index or a list
        is_list = isinstance(data, (list, tuple)) or data.dtype == object

        # Initialize the base class
        if not is_list:
            init_data = data

        elif len(data):
            init_data = data[0]

        else:
            if default is None:
                warn(
                    "The input index data is an empty list without a default "
                    "index. Will use numpy as an underlying representation."
                )
                default = np.empty(0, dtype=np.int64)

            init_data = default

        super().__init__(init_data, is_list=is_list)

        # Get the counts if they are not provided for free
        if counts is None:
            assert (
                batch_ids is not None and batch_size is not None
            ), "Must provide `batch_size` alongside `batch_ids`."
            counts = self.get_counts(batch_ids, batch_size)

        else:
            batch_size = len(counts)

        # Get the number of index elements per entry in the batch
        if single_counts is None:
            assert (
                not self.is_list
            ), "When initializing an index list, provide `single_counts`."
            single_counts = counts
        else:
            assert len(single_counts) == len(
                data
            ), "There must be one single count per index in the list."

        # Cast
        counts = self._as_long(counts)
        single_counts = self._as_long(single_counts)
        offsets = self._as_long(offsets)

        # Do a couple of basic sanity checks
        assert self._sum(counts) == len(
            data
        ), "The `counts` provided must add up to the index length."
        assert len(counts) == len(
            offsets
        ), "The number of `offsets` must match the number of `counts`."

        # Get the boundaries between successive index using the counts
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.single_counts = single_counts
        self.edges = edges
        self.offsets = offsets
        self.batch_size = batch_size

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
        if not self.is_list:
            return self.data[lower:upper] - self.offsets[batch_id]

        else:
            entry = np.empty(upper - lower, dtype=object)
            for i, index in enumerate(self.data[lower:upper]):
                entry[i] = index - self.offsets[batch_id]

            return entry

    @property
    def index(self):
        """Alias for the underlying data stored.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Underlying index
        """
        assert (
            not self.is_list
        ), "Underlying data is not a single index, use `index_list`"

        return self.data

    @property
    def index_list(self):
        """Alias for the underlying data list stored.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            Underlying index list
        """
        assert self.is_list, "Underlying data is a single index, use `index`"

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
    def index_ids(self):
        """Returns the ID of the index in the list each element belongs to.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (M) List of index IDs for each element
        """
        assert self.is_list, "Underlying data must be a list of index"

        return self._repeat(self._arange(len(self.data)), self.single_counts)

    @property
    def full_counts(self):
        """Returns the total number of elements in each batch entry.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (B) Number of elements in each batch entry
        """
        if not self.is_list:
            return self.counts
        else:
            full_counts = self._empty(self.batch_size)
            for b in range(self.batch_size):
                lower, upper = self.edges[b], self.edges[b + 1]
                full_counts[b] = self._sum(self.single_counts[lower:upper])

            return self._as_long(full_counts)

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
        List[List[Union[np.ndarray, torch.Tensor]]]
            List of list of indexes per entry in the batch
        """
        # Cast to numpy object array to be able to use split
        if self.is_list and not isinstance(self.data, np.ndarray):
            data_np = np.empty(len(self.data), dtype=object)
            data_np[:] = self.data
        else:
            data_np = self.data

        # Split, offset
        indexes = np.split(data_np, self.splits)
        for batch_id in range(self.batch_size):
            indexes[batch_id] = indexes[batch_id] - self.offsets[batch_id]

        return indexes

    def merge(self, index_batch):
        """Merge this index batch with another.

        Parameters
        ----------
        index_batch : IndexBatch
            Other index batch object to merge with

        Returns
        -------
        IndexBatch
            Merged index batch
        """
        # Basic cross-checks
        assert (
            self.offsets == index_batch.offsets
        ).all(), "Both index batches should point to the same tensor."

        # Stack the indexes entry-wise in the batch
        indexes, single_counts = [], []
        for b in range(self.batch_size):
            if self.is_list:
                lower, upper = self.edges[b], self.edges[b + 1]
                indexes.extend(self.index_list[lower:upper])
                single_counts.extend(self.single_counts[lower:upper])

                lower, upper = index_batch.edges[b], index_batch.edges[b + 1]
                indexes.extend(index_batch.index_list[lower:upper])
                single_counts.extend(index_batch.single_counts[lower:upper])

            else:
                lower, upper = self.edges[b], self.edges[b + 1]
                indexes.append(self.index[lower:upper])

                lower, upper = index_batch.edges[b], index_batch.edges[b + 1]
                indexes.append(index_batch.index[lower:upper])

        counts = self.counts + index_batch.counts

        if self.is_list:
            return IndexBatch(indexes, self.offsets, counts, single_counts)
        else:
            return IndexBatch(indexes, self.offsets, counts)

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

        if not self.is_list:
            data = self._to_numpy(self.data)
        else:
            data = np.empty(len(self.data), dtype=object)
            for i, d in enumerate(self.data):
                data[i] = self._to_numpy(d)

        offsets = self._to_numpy(self.offsets)
        counts = self._to_numpy(self.counts)

        single_counts = None
        if self.is_list:
            single_counts = self._to_numpy(self.single_counts)

        return IndexBatch(data, offsets, counts, single_counts)

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

        if not self.is_list:
            data = self._to_tensor(self.data, dtype, device)
        else:
            data = np.empty(len(data), dtype=object)
            for i, d in enumerate(self.data):
                data[i] = self._to_tensor(d, dtype, device)

        offsets = self._to_tensor(self.offsets, dtype, device)
        counts = self._to_tensor(self.counts, dtype, device)

        single_counts = None
        if self.is_list:
            single_counts = self._to_tensor(self.single_counts, dtype, device)

        return IndexBatch(data, offsets, counts, single_counts)
