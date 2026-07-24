"""Module with a dataclass targeted at batched matrix/tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from spine.constants import BATCH_COL, COORD_COLS
from spine.utils.conditional import torch

from .base import ArrayLike, BatchBase

__all__ = ["TensorBatch"]


@dataclass(eq=False)
class TensorBatch(BatchBase):
    """Batched tensor with the necessary methods to slice it."""

    data: ArrayLike
    counts: ArrayLike
    edges: ArrayLike
    batch_size: int
    has_batch_col: bool
    coord_cols: Sequence[int] | np.ndarray | None

    def __init__(
        self,
        data: ArrayLike,
        counts: Sequence[int] | ArrayLike | None = None,
        batch_size: int | None = None,
        has_batch_col: bool = False,
        coord_cols: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Initialize the attributes of the class.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            (N, C) Batched tensors
        counts : Union[List[int], np.ndarray, torch.Tensor]
            (B) Number of data rows in each entry
        batch_size : int, optional
            Number of entries that make up the batched data
        has_batch_col : bool, default False
            Wheather the tensor has a column specifying the batch ID
        coord_cols : Union[List[int], np.ndarray], optional
            List of columns specifying coordinates
        """
        # Initialize the base class
        super().__init__(data)

        # Should provide either the counts, or the batch size
        if (counts is not None) == (batch_size is not None):
            raise ValueError("Provide either `counts` or `batch_size`, not both.")

        # If the counts are not provided, must build them once
        if counts is None:
            # Define the array functions depending on the input type
            if not has_batch_col:
                raise ValueError("Cannot get the counts without a batch column.")
            if batch_size is None:  # pragma: no cover
                raise ValueError("Must provide `batch_size` to infer counts.")
            batch_size_value = batch_size

            ref = data
            counts = self.get_counts(ref[:, BATCH_COL], batch_size_value)
        else:
            # If the number of batches is not provided, get it from the counts
            batch_size_value = len(counts)

        # Cast
        counts = self._as_long(counts)
        if self._sum(counts) != len(data):
            raise ValueError(
                "The `counts` provided do not add up to the tensor length."
            )

        # Get the boundaries between entries in the batch
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.edges = edges
        self.batch_size = batch_size_value
        self.has_batch_col = has_batch_col
        self.coord_cols = coord_cols

    def __getitem__(self, batch_id: int) -> ArrayLike:
        """Returns a subset of the tensor corresponding to one entry.

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
        return self.data[lower:upper]

    @property
    def tensor(self) -> ArrayLike:
        """Alias for the underlying data stored.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Underlying tensor of data
        """
        return self.data

    @property
    def batch_ids(self) -> ArrayLike:
        """Returns the batch ID of each of the elements in the tensor.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (N) Batch ID of each element in the tensor
        """
        return self._repeat(self._arange(self.batch_size), self.counts)

    def split(self) -> list[ArrayLike]:
        """Breaks up the tensor batch into its constituents.

        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of one tensor per entry in the batch
        """
        return self._split(self.data, self.splits)

    def apply_mask(self, mask: ArrayLike) -> None:
        """Apply a global mask to the underlying tensor, update batching.

        Parameters
        ----------
        mask : Union[np.ndarray, torch.Tensor]
            (N) Boolean mask to apply to the underlying tensor
        """
        # Update underlying tensor in place
        self.data = self.data[mask]

        # Update batching information
        batch_ids = self.batch_ids[mask]
        self.counts = self.get_counts(batch_ids, self.batch_size)
        self.edges = self.get_edges(self.counts)

    def merge(self, tensor_batch: "TensorBatch") -> "TensorBatch":
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

    def to_numpy(self) -> "TensorBatch":
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
        data = self._to_numpy(data)
        counts = self._to_numpy(self.counts)

        return TensorBatch(
            data, counts, has_batch_col=self.has_batch_col, coord_cols=self.coord_cols
        )

    def to_tensor(self, dtype: Any = None, device: Any = None) -> "TensorBatch":
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

        data = self._to_tensor(self.data, dtype, device)
        counts = self._to_tensor(self.counts, dtype, device)

        return TensorBatch(
            data, counts, has_batch_col=self.has_batch_col, coord_cols=self.coord_cols
        )

    def to_cm(self, meta: Any) -> None:
        """Converts the pixel coordinates of the tensor to cm.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        if not self.is_numpy:
            raise ValueError("Can only convert units of numpy arrays.")
        data = self.data
        data[:, COORD_COLS] = meta.to_cm(data[:, COORD_COLS], center=True)

    def to_px(self, meta: Any) -> None:
        """Converts the coordinates of the tensor to pixel indexes.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        if not self.is_numpy:
            raise ValueError("Can only convert units of numpy arrays.")
        data = self.data
        data[:, COORD_COLS] = meta.to_px(data[:, COORD_COLS], floor=True)

    @classmethod
    def from_list(cls, data_list: Sequence[ArrayLike]) -> "TensorBatch":
        """Builds a batch from a list of tensors.

        Parameters
        ----------
        data_list : List[Union[np.ndarray, torch.Tensor]]
            List of tensors, exactly one per batch
        """
        # Check that we are not fed an empty list of tensors
        if not len(data_list):
            raise ValueError("Must provide at least one tensor to build a tensor batch")
        is_numpy = not isinstance(data_list[0], torch.Tensor)

        # Compute the counts from the input list
        counts = [len(t) for t in data_list]

        # Concatenate input
        if is_numpy:
            return cls(np.concatenate(data_list, axis=0), counts)
        else:
            return cls(torch.cat(data_list, dim=0), counts)
