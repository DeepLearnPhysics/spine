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

from .meta import Meta


@dataclass
class TensorBatch:
    """Batched tensor with the necessary methods to slice it.

    Attributes
    ----------
    tensor : Union[np.ndarray, torch.Tensor, ME.SparseTensor]
        (N, C) Batched tensor where the batch column is `BATCH_COL`
    splits : Union[List, np.ndarray, torch.Tensor]
        (B) Indexes where to split the batch to get its constituents
    offsets : Union[List, np.ndarray, torch.Tensor]
        (B) Offset between successive indexes in the batch. This must only
        be specified if the batched tensor is a simple index.
    batch_size : int
        Number of entries that make up the batched tensor
    """
    tensor: np.ndarray
    splits: np.ndarray
    offsets: np.ndarray
    batch_size: int

    def __init__(self, tensor, splits=None, offsets=None,
                 batch_size=None, sparse=False):
        """Initialize the attributes of the class.

        Parameters
        ----------
        tensor : Union[np.ndarray, torch.Tensor, ME.SparseTensor]
            (N, C) Batched tensor where the batch column is `BATCH_COL`
        splits : Union[List[int], np.ndarray, torch.Tensor], optional
            (B) Indexes where to split the batch to get its constituents
        offsets : Union[List[int], np.ndarray, torch.Tensor], optional
            (B) Offset between successive indexes in the batch. This must only
            be specified if the batched tensor is an index.
        batch_size : int, optional
            Number of entries that make up the batched tensor
        sparse : bool, False
            If initializing from an ME sparse tensor, flip to True
        """
        # Should provide either the split boundaries, or the batch size
        assert (splits is not None) ^ (batch_size is not None), (
                "Provide either `splits` or `batch_size`, not both")

        # Check the typing of the input, store the split function
        self._sparse   = sparse
        self._is_numpy = not sparse and not isinstance(tensor, torch.Tensor)
        self._split_fn = np.split if self._is_numpy else torch.tensor_split

        # If the number of batches is not provided, measure it
        if batch_size is None:
            batch_size = len(splits)

        # If the split boundaries are not provided, must build them once
        if splits is None:
            # Define the array functions depending on the input type
            ref = tensor if not sparse else tensor.C
            splits = self.get_splits(ref[:, BATCH_COL], batch_size)

        # Check that the offsets provided are of the expected length
        if offsets is not None:
            assert len(offsets) == batch_size, (
                    "Should provide one offset per batch ID")

        # Store the attributes
        self.tensor = tensor
        self.splits = splits
        self.offsets = offsets
        self.batch_size = batch_size

    def __len__(self):
        """Returns the number of entries that make up the batch."""
        return self.batch_size

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
        offset = 0 if self.offsets is None else self.offsets[batch_id]
        lower = self.splits[batch_id-1] if batch_id > 0 else 0
        upper = self.splits[batch_id]
        if not self._sparse:
            return self.tensor[lower:upper] - offset
        else:
            from MinkowskiEngine import SparseTensor
            return SparseTensor(
                    self.tensor.F[lower:upper] - offset,
                    coordinates=self.tensor.C[lower:upper])

    def get_splits(self, batch_ids, batch_size):
        """Finds the boundaries between batches, provided a batch ID list.

        Parameters
        ----------
        batch_ids : np.ndarrary
            List of batch IDs
        batch_size : int, optional
            Number of entries that make up the batched tensor

        Returns
        -------
        np.ndarray
            (B) Indexes where to split the batch to get its constituents
        """
        # Define the array functions depending on the input type
        if self._is_numpy:
            zeros = lambda x: np.zeros(x, dtype=np.int64)
            ones = lambda x: np.ones(x, dtype=np.int64)
            unique = lambda x: np.unique(x, return_index=True)
        else:
            zeros = lambda x: torch.zeros(
                    x, dtype=torch.long, device=batch_ids.device)
            ones = lambda x: torch.ones(
                    x, dtype=torch.long, device=batch_ids.device)
            unique = unique_index

        # Get the split list
        if not len(batch_ids):
            # If the tensor is empty, nothing to divide
            return zeros(batch_size)
        else:
            # Find the first index of each batch ID in the input tensor
            uni, index = unique(batch_ids)
            splits = -1 * ones(batch_size)
            splits[uni[:-1]] = index[1:]
            splits[uni[-1]] = len(batch_ids)
            for i, s in enumerate(splits):
                if s < 0:
                    splits[i] = splits[-1] if i > 0 else 0

            return splits

    def split(self):
        """Breaks up the batch into its constituents.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one tensor per entry in the batch
        """
        if not self._sparse:
            return self._split_fn(self.tensor, self.splits[:-1])
        else:
            from MinkowskiEngine import SparseTensor
            coords = self._split_fn(self.tensor.C, self.splits[:-1])
            feat = self._split_fn(self.tensor.F, self.splits[:-1])
            return [SparseTensor(
                feats[i], coordinates=coords[i]) for i in self.batch_size]

    def to_numpy(self):
        """Cast underlying tensor to a `np.ndarray` and return a new instance.

        Returns
        -------
        TensorBatch
            New `TensorBatch` object with an underlying np.ndarray tensor.
        """
        assert not self._is_numpy, (
                "Must be a `torch.Tensor` to be cast to `np.ndarray`")

        tensor = self.tensor
        if self._sparse:
            tensor = torch.cat([self.tensor.C.float(), self.tensor.F], dim=1)

        tensor = tensor.cpu().detach().numpy()
        splits = self.splits.cpu().detach().numpy()

        return TensorBatch(tensor, splits)

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
        assert self._is_numpy, (
                "Must be a `np.ndarray` to be cast to `torch.Tensor`")

        tensor = torch.as_tensor(self.tensor, dtype=dtype, device=device)
        splits = torch.as_tensor(self.splits, dtype=torch.int64, device=device)
        return TensorBatch(tensor, splits)

    def to_cm(self, meta):
        """Converts the coordinates of the tensor to cm.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self._is_numpy, "Can only convert units of numpy arrays"
        self.tensor[:, COORD_COLS] = meta.to_cm(self.tensor[:, COORD_COLS])

    def to_pixel(self, meta):
        """Converts the coordinates of the tensor to pixel indexes.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self._is_numpy, "Can only convert units of numpy arrays"
        self.tensor[:, COORD_COLS] = meta.to_pixel(self.tensor[:, COORD_COLS])

    @classmethod
    def from_list(cls, tensor_list):
        """Builds a batch from a list of tensors.

        Parameters
        ----------
        tensor_list : List[Union[np.ndarray, torch.Tensor]]
            List of tensors, exactly one per batch
        """
        # Check that we are not fed an empty list of tensors
        assert len(tensor_list), (
                "Must provide at least one tensor to build a tensor batch")
        is_numpy = not isinstance(tensor_list[0], torch.Tensor)

        # Compute the splits from the input list
        counts = [len(t) for t in tensor_list]
        splits = np.cumsum(counts)
        if not is_numpy:
            device = tensor_list[0].device
            splits = torch.as_tensor(splits, dtype=torch.int64, device=device)

        # Concatenate input
        if is_numpy:
            return cls(np.concatenate(tensor_list, axis=0), splits)
        else:
            return cls(torch.cat(tensor_list, axis=0), splits)

@dataclass
class IndexBatch:
    """Batched index with the necessary methods to slice it.
    
    Attributes
    ----------
    TODO
    """
    index: np.ndarray
    splits: np.ndarray
    offsets: np.ndarray
    entry_splits: np.ndarray


