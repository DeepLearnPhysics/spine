"""Module with a dataclass targeted at batched matrix/tensors."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Union, List

from mlreco.data_structures.meta import Meta
from mlreco.utils.globals import BATCH_COL, COORD_COLS
from mlreco.utils.decorators import inherit_docstring

from .base import BatchBase

__all__ = ['TensorBatch']


@dataclass
@inherit_docstring(BatchBase)
class TensorBatch(BatchBase):
    """Batched tensor with the necessary methods to slice it."""

    def __init__(self, data, counts=None, batch_size=None, is_sparse=False, 
                 has_batch_col=None, coord_cols=None):
        """Initialize the attributes of the class.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor, ME.SparseTensor]
            (N, C) Batched data where the batch column is `BATCH_COL`
        counts : Union[List[int], np.ndarray, torch.Tensor]
            (B) Number of data rows in each entry
        batch_size : int, optional
            Number of entries that make up the batched data
        is_sparse : bool, default False
            If initializing from an ME sparse data, flip to True
        has_batch_col : bool, default False
            Column specifying the batch ID of each row
        coord_cols : Union[List[int], np.ndarray], optional
            List of columns specifying coordinates
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

        # If the data is sparse, it must have a batch column and coordinates
        if is_sparse:
            has_batch_col = True
            coord_cols = COORD_COLS

        # Get the boundaries between entries in the batch
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.edges = edges
        self.batch_size = batch_size
        self.has_batch_col = has_batch_col
        self.coord_cols = coord_cols

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

    def merge(self, tensor_batch):
        """Merge this tensor batch with another.

        Parameters
        ----------
        tensor_batch : TensorBatch
            Other tensor batch object to merge with

        Returns
        -------
        TensorBatch
            Merged tensor batch
        """
        # Stack the tensors entry-wise in the batch
        entries = []
        for b in range(self.batch_size):
            entries.append(self[b])
            entries.append(tensor_batch[b])

        tensor = self._cat(entries)
        counts = self.counts + tensor_batch.counts

        return TensorBatch(tensor, counts)

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
