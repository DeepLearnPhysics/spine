"""Self-describing batched tensor products."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..base import TensorSchema
from ..tensor import TensorData

if TYPE_CHECKING:  # pragma: no cover
    import torch
else:
    from spine.utils.conditional import torch

from .base import ArrayLike, BatchBase

__all__ = ["TensorBatch"]


@dataclass(eq=False)
class TensorBatch(BatchBase):
    """Concatenated tensor rows with event boundaries and a logical schema.

    ``counts`` and ``edges`` describe the event partition without requiring a
    packed batch column. When a batch column is retained for a sparse backend,
    it is excluded from logical feature access. Coordinate and feature names
    remain relative to their separate logical matrices.

    Attributes
    ----------
    data : numpy.ndarray or torch.Tensor
        Concatenated rows for every event.
    counts : numpy.ndarray or torch.Tensor
        Number of rows per event.
    edges : numpy.ndarray or torch.Tensor
        Cumulative event boundaries.
    batch_size : int
        Number of events represented.
    has_batch_col : bool
        Whether packed column zero stores batch IDs.
    coord_cols : sequence[int], optional
        Packed columns containing every coordinate group.
    schema : TensorSchema
        Logical coordinate, feature and overlay metadata.
    meta : list, optional
        Event-aligned metadata.
    """

    data: ArrayLike
    counts: ArrayLike
    edges: ArrayLike
    batch_size: int
    has_batch_col: bool
    coord_cols: Sequence[int] | np.ndarray | None
    schema: TensorSchema
    meta: list[Any] | None

    def __init__(
        self,
        data: ArrayLike,
        counts: Sequence[int] | ArrayLike | None = None,
        batch_size: int | None = None,
        has_batch_col: bool = False,
        coord_cols: Sequence[int] | np.ndarray | None = None,
        schema: TensorSchema | None = None,
        meta: list[Any] | None = None,
    ) -> None:
        """Initialize a batched tensor from counts or packed batch IDs.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            (N, C) Batched tensors
        counts : Union[List[int], np.ndarray, torch.Tensor], optional
            (B) Number of data rows in each entry
        batch_size : int, optional
            Number of entries that make up the batched data
        has_batch_col : bool, default False
            Whether the tensor has a column specifying the batch ID
        coord_cols : Union[List[int], np.ndarray], optional
            List of columns specifying coordinates
        schema : TensorSchema, optional
            Logical coordinate and feature description
        meta : list, optional
            Per-event metadata retained during batching and unwrapping

        Raises
        ------
        ValueError
            If count specification is ambiguous, inferred counts lack a batch
            column, or counts do not cover every row.
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
            counts = self.get_counts(ref[:, 0], batch_size_value)
        else:
            # If the number of batches is not provided, get it from the counts
            batch_size_value = len(counts)

        # Normalize counts and derive event boundaries
        counts = self._as_long(counts)
        if self._sum(counts) != len(data):
            raise ValueError(
                "The `counts` provided do not add up to the tensor length."
            )

        # Get the boundaries between entries in the batch
        edges = self.get_edges(counts)

        # Store the physical batch description
        self.data = data
        self.counts = counts
        self.edges = edges
        self.batch_size = batch_size_value
        self.has_batch_col = has_batch_col
        self.coord_cols = coord_cols

        # Infer a logical schema only when the producer did not provide one
        if schema is None:
            width = 0 if coord_cols is None else len(coord_cols)
            coords = None if not width else np.empty((0, width), dtype=np.float32)
            feature_width = data.shape[1] if data.ndim > 1 else 1
            feature_width -= width + int(has_batch_col)
            features = np.empty((0, feature_width), dtype=np.float32)
            schema = TensorSchema.infer(
                coords,
                features=features,
                feats_only=coord_cols is None,
            )
        self.schema = schema
        self.meta = meta

    @property
    def coordinate_groups(self) -> dict[str, tuple[int, ...]]:
        """Return named coordinate groups defined by the product schema."""
        return self.schema.coordinate_groups

    def coordinate_columns(self, name: str | None = None) -> np.ndarray:
        """Resolve one semantic coordinate group to packed tensor columns.

        Parameters
        ----------
        name : str, optional
            Coordinate group. It may be omitted for a single-group schema.

        Returns
        -------
        numpy.ndarray
            Packed column indexes for the requested group.
        """
        if self.coord_cols is None:
            raise ValueError("This tensor batch has no coordinate columns.")

        # Resolve an omitted group only when the schema has a unique answer
        if name is None:
            if len(self.coordinate_groups) != 1:
                raise ValueError(
                    "Coordinate group is ambiguous; specify one of "
                    f"{tuple(self.coordinate_groups)}."
                )
            name = next(iter(self.coordinate_groups))
        if name not in self.coordinate_groups:
            raise KeyError(f"Unknown coordinate group `{name}`.")

        # Map schema-relative positions onto the packed tensor columns
        return np.asarray(self.coord_cols)[list(self.coordinate_groups[name])]

    def feature_columns(self, name: str | None = None) -> np.ndarray:
        """Resolve all features or one named field to packed tensor columns.

        Batch and coordinate columns are excluded before applying the
        feature-relative positions stored in the schema.
        """
        if self.data.ndim == 1:
            if name is not None and name not in self.schema.feature_fields:
                raise KeyError(f"Unknown feature field `{name}`.")
            return np.asarray([0], dtype=np.int64)

        # Identify physical feature columns by excluding batch and coordinates
        excluded = set(() if self.coord_cols is None else self.coord_cols)
        if self.has_batch_col:
            excluded.add(self.batch_col)
        columns = np.asarray(
            [column for column in range(self.data.shape[1]) if column not in excluded],
            dtype=np.int64,
        )
        if name is None:
            return columns

        # Project a named logical field onto those physical feature columns
        if name not in self.schema.feature_fields:
            raise KeyError(f"Unknown feature field `{name}`.")

        return columns[list(self.schema.feature_fields[name])]

    def coordinates(self, name: str | None = None) -> "TensorBatch":
        """Return one named coordinate group as a tensor batch.

        When a product advertises multiple coordinate groups, such as start
        and end points, callers must select one explicitly.

        Parameters
        ----------
        name : str, optional
            Requested coordinate group.

        Returns
        -------
        TensorBatch
            Coordinate-only batch with the original event counts.
        """
        # Slice the requested group, then describe the smaller coordinate space
        columns = self.coordinate_columns(name)
        values = self.data[:, columns]

        return TensorBatch(
            values,
            self.counts,
            coord_cols=np.arange(len(columns), dtype=np.int64),
            schema=TensorSchema.infer(np.empty((0, len(columns)))),
        )

    @property
    def coords(self) -> "TensorBatch":
        """Return the sole coordinate group as a tensor batch."""
        return self.coordinates()

    @property
    def coordinate_data(self) -> "TensorBatch | None":
        """Return all coordinate columns without group disambiguation.

        This interface is intended for infrastructure which must transform or
        serialize every coordinate group at once.
        """
        if self.coord_cols is None:
            return None

        # Retain every coordinate group while dropping packed non-coordinates
        columns = np.asarray(self.coord_cols)

        return TensorBatch(
            self.data[:, columns],
            self.counts,
            coord_cols=np.arange(len(columns), dtype=np.int64),
            schema=self.schema,
            meta=self.meta,
        )

    @property
    def batch_coordinates(self) -> ArrayLike:
        """Return coordinates prefixed by their batch IDs.

        Sparse convolution backends use this packed representation at their
        boundary. Logical consumers should use :attr:`coords` instead.
        """
        coordinates = self.coordinate_data
        if coordinates is None:
            raise ValueError("This tensor batch has no coordinate columns.")

        # Reintroduce batch IDs only at the sparse-backend boundary
        values = coordinates.data
        batch_ids = self.batch_ids
        if isinstance(values, np.ndarray):
            return np.concatenate((batch_ids[:, None], values), axis=1)

        return torch.cat((batch_ids[:, None], values), dim=1)

    @property
    def features(self) -> "TensorBatch":
        """Return all non-batch, non-coordinate columns as a new batch."""
        if self.data.ndim == 1:
            return TensorBatch(self.data, self.counts, schema=self.schema)
        columns = self.feature_columns()
        return TensorBatch(self.data[:, columns], self.counts, schema=TensorSchema())

    def feature(self, field: str | int) -> "TensorBatch":
        """Return one named or positional feature field.

        Parameters
        ----------
        field : str or int
            Schema field name or position in the logical feature matrix.
        """
        # Resolve the named or positional selection against logical features
        features = self.features
        if isinstance(field, str):
            columns = self.schema.feature_fields[field]
            if features.data.ndim == 1:
                if columns != (0,):
                    raise IndexError(
                        "One-dimensional feature products only contain column 0."
                    )
                values = features.data
            else:
                values = features.data[:, columns]
        else:
            if features.data.ndim == 1:
                if field not in (0, -1):
                    raise IndexError(
                        "One-dimensional feature products only contain column 0."
                    )
                values = features.data
            else:
                values = features.data[:, field]

        return TensorBatch(values, self.counts, schema=TensorSchema())

    @property
    def values(self) -> "TensorBatch":
        """Return the primary feature as a one-dimensional batch."""
        features = self.features
        if features.data.ndim == 1:
            return features
        if features.data.ndim != 2 or features.data.shape[1] == 0:
            raise ValueError("`values` requires at least one feature column.")

        # Keep the primary charge/value convention independent of any
        # auxiliary point features carried beside it.
        return TensorBatch(features.data[:, 0], self.counts, schema=TensorSchema())

    def event(self, batch_id: int) -> TensorData:
        """Return one event while preserving schema and metadata.

        Parameters
        ----------
        batch_id : int
            Event position in the batch.

        Returns
        -------
        TensorData
            Batch-free product with coordinates and features separated.
        """
        # Extract physical rows and the event-aligned auxiliary metadata
        data = self[batch_id]
        coord_cols = np.asarray(
            () if self.coord_cols is None else self.coord_cols, dtype=np.int64
        )
        meta = None if self.meta is None else self.meta[batch_id]

        # Feature-only vectors need no packed-column decomposition
        if data.ndim == 1:
            if len(coord_cols):
                raise ValueError(
                    "One-dimensional tensor products cannot carry coordinates."
                )
            return TensorData(data, meta=meta, schema=self.schema)

        # Remove batch and coordinate columns from the event feature matrix
        excluded = set(coord_cols.tolist())
        if self.has_batch_col:
            excluded.add(self.batch_col)
        feature_cols = [
            column for column in range(data.shape[1]) if column not in excluded
        ]
        coords = None if not len(coord_cols) else data[:, coord_cols]
        features = data[:, feature_cols]

        return TensorData(features, coords, meta=meta, schema=self.schema)

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

    def numpy_tensor(self) -> np.ndarray:
        """Return the underlying data narrowed to a NumPy array.

        Raises
        ------
        TypeError
            If this batch is backed by a PyTorch tensor.
        """
        if not isinstance(self.data, np.ndarray):
            raise TypeError("TensorBatch is not backed by a numpy.ndarray.")
        return self.data

    def torch_tensor(self) -> torch.Tensor:
        """Return the underlying data narrowed to a PyTorch tensor.

        Raises
        ------
        TypeError
            If this batch is backed by a NumPy array.
        """
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("TensorBatch is not backed by a torch.Tensor.")
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

    @property
    def batch_col(self) -> int:
        """Return the packed batch-ID column used at sparse boundaries."""
        if not self.has_batch_col:
            raise ValueError("This tensor batch has no packed batch column.")
        return 0

    def split(self) -> list[ArrayLike]:
        """Break the batch into packed arrays, one per event.

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

    def select(self, mask: ArrayLike) -> "TensorBatch":
        """Return selected rows while preserving schema and event metadata.

        Parameters
        ----------
        mask : numpy.ndarray or torch.Tensor
            Global boolean mask or row indexes. Rows must remain grouped in
            event order.
        """
        data = self.data[mask]
        batch_ids = self.batch_ids[mask]
        counts = self.get_counts(batch_ids, self.batch_size)
        return TensorBatch(
            data,
            counts,
            has_batch_col=self.has_batch_col,
            coord_cols=self.coord_cols,
            schema=self.schema,
            meta=self.meta,
        )

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

        # Logical descriptions must agree even when physical widths happen to match
        if self.schema != tensor_batch.schema:
            raise ValueError("Cannot merge tensor batches with different schemas.")

        # Preserve metadata only when both inputs describe the same event axis
        meta = None
        if self.meta is not None and tensor_batch.meta is not None:
            meta = self.meta
        return TensorBatch(
            tensor,
            counts,
            has_batch_col=self.has_batch_col,
            coord_cols=self.coord_cols,
            schema=self.schema,
            meta=meta,
        )

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
            data,
            counts,
            has_batch_col=self.has_batch_col,
            coord_cols=self.coord_cols,
            schema=self.schema,
            meta=self.meta,
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
            data,
            counts,
            has_batch_col=self.has_batch_col,
            coord_cols=self.coord_cols,
            schema=self.schema,
            meta=self.meta,
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
        if self.coord_cols is None:
            raise ValueError("Cannot convert a tensor without coordinate metadata.")
        data = self.data
        for group in self.coordinate_groups.values():
            columns = np.asarray(self.coord_cols)[list(group)]
            data[:, columns] = meta.to_cm(data[:, columns], center=True)

    def to_px(self, meta: Any) -> None:
        """Converts the coordinates of the tensor to pixel indexes.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        if not self.is_numpy:
            raise ValueError("Can only convert units of numpy arrays.")
        if self.coord_cols is None:
            raise ValueError("Cannot convert a tensor without coordinate metadata.")
        data = self.data
        for group in self.coordinate_groups.values():
            columns = np.asarray(self.coord_cols)[list(group)]
            data[:, columns] = meta.to_px(data[:, columns], floor=True)

    @classmethod
    def from_list(cls, data_list: Sequence[ArrayLike]) -> "TensorBatch":
        """Build a feature-only batch from raw event arrays.

        Parameters
        ----------
        data_list : List[Union[np.ndarray, torch.Tensor]]
            List of tensors, exactly one per batch

        Returns
        -------
        TensorBatch
            Concatenated feature-only batch.
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

    @classmethod
    def from_data_list(cls, data_list: Sequence[TensorData]) -> "TensorBatch":
        """Build a batch from self-describing event tensor products.

        Parameters
        ----------
        data_list : sequence[TensorData]
            Event products with an identical schema and storage backend.

        Returns
        -------
        TensorBatch
            Packed batch which preserves coordinates, schema and metadata.
        """
        if not len(data_list):
            raise ValueError("Must provide at least one event tensor product.")
        reference = data_list[0]
        if any(data.schema != reference.schema for data in data_list[1:]):
            raise ValueError("Cannot batch event tensors with different schemas.")
        if any(
            (data.coordinate_data is None) != (reference.coordinate_data is None)
            for data in data_list[1:]
        ):
            raise ValueError("Coordinate presence must be consistent across a batch.")

        # Do not silently copy values between NumPy and PyTorch backends
        is_numpy = isinstance(reference.features, np.ndarray)
        if any(
            isinstance(data.features, np.ndarray) != is_numpy for data in data_list[1:]
        ):
            raise ValueError("Event tensors in one batch must share an array backend.")

        # Concatenate the logical feature matrices first
        counts = [len(data) for data in data_list]
        cat = (
            np.concatenate
            if is_numpy
            else lambda arrays, axis=0: torch.cat(arrays, dim=axis)
        )
        features = cat([data.features for data in data_list], axis=0)
        if reference.coordinate_data is None:
            return cls(
                features,
                counts,
                schema=reference.schema,
                meta=[data.meta for data in data_list],
            )

        # Coordinate-bearing batches use the sparse-compatible packed layout
        coords = cat([data.coordinate_data for data in data_list], axis=0)
        if is_numpy:
            batch_ids = np.repeat(np.arange(len(data_list)), counts)
        else:
            batch_ids = torch.repeat_interleave(
                torch.arange(len(data_list), device=coords.device),
                torch.as_tensor(counts, device=coords.device),
            )

        # Normalize scalar arrays before assembling the sparse-compatible table
        if coords.ndim == 1:
            coords = coords[:, None]
        if features.ndim == 1:
            features = features[:, None]

        packed = cat((batch_ids[:, None], coords, features), axis=1)
        coord_cols = np.arange(1, 1 + coords.shape[1], dtype=np.int64)

        return cls(
            packed,
            counts,
            has_batch_col=True,
            coord_cols=coord_cols,
            schema=reference.schema,
            meta=[data.meta for data in data_list],
        )
