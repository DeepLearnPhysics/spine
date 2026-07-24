"""Backend-neutral sparse tensor exposed to SPINE models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from spine.data.batch import TensorBatch

from . import backend

__all__ = ["SparseTensor"]


def _normalize_stride(
    stride: int | Sequence[int] | torch.Tensor, dimension: int
) -> tuple[int, ...]:
    """Normalize a tensor stride to one value per spatial dimension.

    Parameters
    ----------
    stride : int, sequence of int or torch.Tensor
        Scalar stride or one stride value per spatial dimension.
    dimension : int
        Number of spatial dimensions.

    Returns
    -------
    tuple of int
        Normalized stride with ``dimension`` entries.
    """
    if isinstance(stride, torch.Tensor):
        stride = stride.detach().cpu().tolist()
    if isinstance(stride, int):
        return (stride,) * dimension
    return tuple(int(value) for value in stride)


class SparseTensor:
    """Sparse model-runtime tensor.

    The public object owns batching and input-row provenance while the selected
    backend owns the active coordinate map. Duplicate input coordinates are
    coalesced for sparse convolution and can be restored at row-aligned output
    boundaries with :meth:`aligned_features`.

    Parameters
    ----------
    features : torch.Tensor, optional
        ``(N, C)`` feature matrix. A one-dimensional input is promoted to
        ``(N, 1)``.
    coordinates : torch.Tensor, optional
        ``(N, D + 1)`` coordinate matrix with the batch index in column zero.
        Either this argument or both ``coordinate_map_key`` and
        ``coordinate_manager`` must be provided.
    tensor_stride : int, sequence of int or torch.Tensor, default 1
        Spatial stride represented by the input coordinates.
    coordinate_map_key : Any, optional
        Existing backend coordinate-map key used when replacing features.
    coordinate_manager : Any, optional
        Existing backend coordinate manager paired with
        ``coordinate_map_key``.
    feats : torch.Tensor, optional
        Compatibility alias for ``features``.
    coords : torch.Tensor, optional
        Compatibility alias for ``coordinates``.
    coords_key : Any, optional
        Compatibility alias for ``coordinate_map_key``.
    coords_manager : Any, optional
        Compatibility alias for ``coordinate_manager``.
    batch_size : int, optional
        Number of batch entries. This must be supplied to preserve trailing
        empty entries that cannot be inferred from coordinates.
    duplicate_reduction : {"sum", "mean", "first"}, default "sum"
        Reduction the backend applies to features with identical coordinates.
    source : SparseTensor, optional
        Tensor whose input-row provenance should be propagated.
    **kwargs : Any
        Additional arguments forwarded to the backend tensor constructor.

    Notes
    -----
    Sparse backends require one feature vector per active coordinate. The
    selected backend therefore coalesces duplicate coordinates while SPINE
    retains original-row provenance only when quantization reduces the input
    length. Calling :meth:`aligned_features` or setting ``restore=True`` in
    :meth:`to_tensor_batch` repeats processed voxel features in the original
    input order. It does not recover distinctions discarded by the duplicate
    reduction. Unique inputs stay on the backend's native fast path.

    Examples
    --------
    Duplicate feature rows are summed internally but remain recoverable as
    logical output rows:

    >>> coordinates = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
    >>> features = torch.tensor([[2.0], [3.0]])
    >>> tensor = SparseTensor(features, coordinates)
    >>> tensor.F
    tensor([[5.]])
    >>> tensor.aligned_features()
    tensor([[5.],
            [5.]])
    """

    def __init__(
        self,
        features: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        tensor_stride: int | Sequence[int] | torch.Tensor = 1,
        coordinate_map_key: Any = None,
        coordinate_manager: Any = None,
        *,
        feats: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        coords_key: Any = None,
        coords_manager: Any = None,
        batch_size: int | None = None,
        duplicate_reduction: str = "sum",
        source: "SparseTensor | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the sparse tensor and its input-row provenance."""
        if features is None:
            features = feats
        if coordinates is None:
            coordinates = coords
        if coordinate_map_key is None:
            coordinate_map_key = coords_key
        if coordinate_manager is None:
            coordinate_manager = coords_manager
        if features is None:
            raise ValueError("SparseTensor requires `features`.")
        if features.ndim == 1:
            features = features[:, None]

        self._backend_tensor = None
        self._batch_size = batch_size
        self._reference_coordinates = None
        self._reference_counts = None
        self._unique_index = None
        self._inverse_index = None

        if coordinates is not None:
            if coordinates.ndim != 2 or features.ndim != 2:
                raise ValueError("Sparse coordinates and features must be matrices.")
            if len(coordinates) != len(features):
                raise ValueError(
                    "Sparse coordinates and features must have equal length."
                )
            if duplicate_reduction not in {"mean", "sum", "first"}:
                raise ValueError(
                    f"Unknown duplicate reduction `{duplicate_reduction}`. "
                    "Choose from 'mean', 'sum' or 'first'."
                )

            dimension = coordinates.shape[1] - 1
            self._tensor_stride = _normalize_stride(tensor_stride, dimension)
            if batch_size is None:
                batch_size = (
                    int(coordinates[:, 0].max().item()) + 1 if len(coordinates) else 0
                )
            self._batch_size = batch_size
            self._coordinates = coordinates
            self._features = features
        else:
            if coordinate_map_key is None or coordinate_manager is None:
                raise ValueError(
                    "Provide coordinates or both a coordinate map key and manager."
                )
            self._features = features
            self._coordinates = None
            dimension = len(coordinate_map_key.get_tensor_stride())
            self._tensor_stride = _normalize_stride(
                coordinate_map_key.get_tensor_stride(), dimension
            )

        if len(self._features):
            if coordinates is not None:
                self._backend_tensor = backend.create_tensor(
                    features=self._features,
                    coordinates=self._coordinates,
                    tensor_stride=self._tensor_stride,
                    coordinate_manager=coordinate_manager,
                    duplicate_reduction=duplicate_reduction,
                    **kwargs,
                )
                self._features = backend.features(self._backend_tensor)
                self._coordinates = backend.coordinates(self._backend_tensor)
                self._unique_index = backend.unique_index(self._backend_tensor)
                self._inverse_index = backend.inverse_mapping(self._backend_tensor)
                if len(self._features) != len(features):
                    self._reference_coordinates = coordinates
                    self._reference_counts = self._counts_from_coordinates(
                        coordinates, self._batch_size
                    )
            else:
                self._backend_tensor = backend.create_tensor(
                    features=self._features,
                    coordinate_map_key=coordinate_map_key,
                    coordinate_manager=coordinate_manager,
                    **kwargs,
                )
        elif coordinates is not None:
            identity = torch.arange(0, device=features.device)
            self._unique_index = identity
            self._inverse_index = identity

        if source is not None:
            self._batch_size = source.batch_size
            self._reference_coordinates = source._reference_coordinates
            self._reference_counts = source._reference_counts
            self._unique_index = source._unique_index
            self._inverse_index = source._inverse_index

    @classmethod
    def from_backend(
        cls, backend_tensor: Any, source: "SparseTensor | None" = None
    ) -> "SparseTensor":
        """Wrap a backend result and propagate SPINE metadata.

        Parameters
        ----------
        backend_tensor : Any
            Native sparse tensor returned by a backend operation.
        source : SparseTensor, optional
            Input tensor from which batch and row-provenance metadata should
            be inherited.

        Returns
        -------
        SparseTensor
            Backend-neutral wrapper around ``backend_tensor``.
        """
        obj = cls.__new__(cls)
        obj._backend_tensor = backend_tensor
        obj._features = backend.features(backend_tensor)
        obj._coordinates = backend.coordinates(backend_tensor)
        obj._tensor_stride = backend.tensor_stride(backend_tensor)
        obj._batch_size = source.batch_size if source is not None else None
        obj._reference_coordinates = (
            source._reference_coordinates if source is not None else None
        )
        obj._reference_counts = source._reference_counts if source is not None else None
        obj._unique_index = source._unique_index if source is not None else None
        obj._inverse_index = source._inverse_index if source is not None else None
        return obj

    @classmethod
    def empty_like(
        cls,
        source: "SparseTensor",
        channels: int | None = None,
        tensor_stride: int | Sequence[int] | torch.Tensor | None = None,
    ) -> "SparseTensor":
        """Create an empty result without entering the sparse backend.

        Parameters
        ----------
        source : SparseTensor
            Tensor supplying device, dtype, coordinates, and provenance.
        channels : int, optional
            Output feature count. By default, preserve the source count.
        tensor_stride : int, sequence of int or torch.Tensor, optional
            Output spatial stride. By default, preserve the source stride.

        Returns
        -------
        SparseTensor
            Empty tensor carrying the requested output metadata.
        """
        obj = cls.__new__(cls)
        num_channels = source.F.shape[1] if channels is None else channels
        obj._backend_tensor = None
        obj._features = source.F.new_empty((0, num_channels))
        obj._coordinates = source.C.new_empty((0, source.C.shape[1]))
        obj._tensor_stride = _normalize_stride(
            source.tensor_stride if tensor_stride is None else tensor_stride,
            source.dimension,
        )
        obj._batch_size = source.batch_size
        obj._reference_coordinates = source._reference_coordinates
        obj._reference_counts = source._reference_counts
        obj._unique_index = source._unique_index
        obj._inverse_index = source._inverse_index
        return obj

    @staticmethod
    def _counts_from_coordinates(
        coordinates: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Count active coordinate rows in each batch entry."""
        if not len(coordinates):
            return torch.zeros(batch_size, dtype=torch.long, device=coordinates.device)
        return torch.bincount(coordinates[:, 0].long(), minlength=batch_size).to(
            coordinates.device
        )

    @property
    def F(self) -> torch.Tensor:
        """Return the ``(N, C)`` sparse feature matrix."""
        return self._features

    @property
    def C(self) -> torch.Tensor:
        """Return the ``(N, D + 1)`` batched coordinate matrix."""
        if self._coordinates is None and self._backend_tensor is not None:
            self._coordinates = backend.coordinates(self._backend_tensor)
        return self._coordinates

    features = F
    coordinates = C
    feats = F
    coords = C

    @property
    def dtype(self) -> torch.dtype:
        """Return the feature data type."""
        return self.F.dtype

    @property
    def device(self) -> torch.device:
        """Return the device holding the feature matrix."""
        return self.F.device

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the feature matrix."""
        return self.F.shape

    @property
    def dimension(self) -> int:
        """Return the number of spatial coordinate dimensions."""
        return self.C.shape[1] - 1

    @property
    def tensor_stride(self) -> tuple[int, ...]:
        """Return the stride represented by each spatial coordinate axis."""
        return self._tensor_stride

    @property
    def batch_size(self) -> int:
        """Return the number of batch entries, including trailing empty ones."""
        if self._batch_size is not None:
            return self._batch_size
        return int(self.C[:, 0].max().item()) + 1 if len(self) else 0

    @property
    def counts(self) -> torch.Tensor:
        """Return the number of active sparse sites in each batch entry."""
        return self._counts_from_coordinates(self.C, self.batch_size)

    @property
    def decomposed_coordinates(self) -> list[torch.Tensor]:
        """Spatial coordinates separated by batch, including empty entries."""
        return [self.coordinates_at(batch_id) for batch_id in range(self.batch_size)]

    @property
    def decomposed_features(self) -> list[torch.Tensor]:
        """Features separated by batch, including empty entries."""
        return [self.features_at(batch_id) for batch_id in range(self.batch_size)]

    @property
    def decomposed_coordinates_and_features(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return per-entry coordinates and features as parallel lists."""
        return self.decomposed_coordinates, self.decomposed_features

    @property
    def coordinate_map_key(self) -> Any:
        """Return the backend coordinate-map key, or ``None`` when empty."""
        if self._backend_tensor is None:
            return None
        return backend.coordinate_map_key(self._backend_tensor)

    @property
    def coordinate_manager(self) -> Any:
        """Return the backend coordinate manager, or ``None`` when empty."""
        if self._backend_tensor is None:
            return None
        return backend.coordinate_manager(self._backend_tensor)

    coords_key = coordinate_map_key
    coords_man = coordinate_manager

    @property
    def backend_tensor(self) -> Any:
        """Return the native tensor consumed by backend-selected modules.

        Returns
        -------
        Any
            Native backend tensor, or ``None`` when there are no active sites.

        Notes
        -----
        Model code should not depend on this property. It exists for operation
        wrappers and backend integration.
        """
        return self._backend_tensor

    @property
    def unique_index(self) -> torch.Tensor | None:
        """Return first input-row indices for the coalesced coordinates."""
        return self._unique_index

    @property
    def inverse_mapping(self) -> torch.Tensor | None:
        """Map each original input row to its coalesced coordinate."""
        return self._inverse_index

    @property
    def reference_size(self) -> int:
        """Number of logical rows represented by the input reference."""
        if self._reference_coordinates is None:
            return len(self)
        return len(self._reference_coordinates)

    def __len__(self) -> int:
        """Return the number of active, unique sparse sites."""
        return len(self.F)

    def __getattr__(self, name: str) -> Any:
        """Delegate backend-specific read access during the migration period."""
        if name.startswith("_") or self._backend_tensor is None:
            raise AttributeError(name)
        return getattr(self._backend_tensor, name)

    def features_at(self, batch_id: int) -> torch.Tensor:
        """Return active features belonging to one batch entry.

        Parameters
        ----------
        batch_id : int
            Batch entry to select.

        Returns
        -------
        torch.Tensor
            ``(N_i, C)`` feature matrix for the selected entry.
        """
        return self.F[self.C[:, 0].long() == batch_id]

    def coordinates_at(self, batch_id: int) -> torch.Tensor:
        """Return spatial coordinates belonging to one batch entry.

        Parameters
        ----------
        batch_id : int
            Batch entry to select.

        Returns
        -------
        torch.Tensor
            ``(N_i, D)`` spatial coordinates without the batch column.
        """
        mask = self.C[:, 0].long() == batch_id
        return self.C[mask, 1:]

    def _wrap(self, backend_tensor: Any) -> "SparseTensor":
        """Wrap a backend result and inherit this tensor's provenance."""
        return SparseTensor.from_backend(backend_tensor, self)

    def replace_features(self, features: torch.Tensor) -> "SparseTensor":
        """Return a tensor on the same coordinate map with new features.

        Parameters
        ----------
        features : torch.Tensor
            ``(N, C_out)`` feature matrix aligned with the active coordinates.

        Returns
        -------
        SparseTensor
            Tensor sharing coordinates and provenance with this tensor.
        """
        if not len(self):
            result = SparseTensor.empty_like(self, features.shape[1])
            result._features = features
            return result
        backend_tensor = backend.create_tensor(
            features=features,
            coordinate_map_key=self.coordinate_map_key,
            coordinate_manager=self.coordinate_manager,
        )
        return self._wrap(backend_tensor)

    def aligned_features(self) -> torch.Tensor:
        """Restore features to the original input rows.

        Returns
        -------
        torch.Tensor
            Feature matrix in the original input order and with the original
            row multiplicity. Duplicate rows receive the same processed sparse
            feature vector.
        """
        reference = self._reference_coordinates
        if reference is None:
            return self.F
        if not len(self):
            return self.F.new_zeros((len(reference), self.F.shape[1]))
        queries = reference.to(device=self.C.device, dtype=self.F.dtype)
        return backend.features_at_coordinates(self.backend_tensor, queries)

    def to_tensor_batch(
        self, *, include_coordinates: bool = True, restore: bool = False
    ) -> TensorBatch:
        """Convert sparse model data to a portable tensor batch.

        Parameters
        ----------
        include_coordinates : bool, default True
            Prefix each output feature row with its batched coordinates.
        restore : bool, default False
            Restore original input ordering and duplicate multiplicity.

        Returns
        -------
        TensorBatch
            Dense batched representation suitable for output handling and
            unwrapping.
        """
        if restore and self._reference_coordinates is not None:
            features = self.aligned_features()
            coordinates = self._reference_coordinates
            counts = self._reference_counts
        else:
            features = self.F
            coordinates = self.C
            counts = self.counts

        if not include_coordinates:
            return TensorBatch(features, counts)

        data = torch.cat([coordinates.to(features.dtype), features], dim=1)
        return TensorBatch(
            data,
            counts,
            has_batch_col=True,
            coord_cols=tuple(range(1, coordinates.shape[1])),
        )

    def detach(self) -> "SparseTensor":
        """Return a tensor whose features are detached from autograd.

        Returns
        -------
        SparseTensor
            Tensor sharing coordinates and provenance with detached features.
        """
        return self.replace_features(self.F.detach())

    def float(self) -> "SparseTensor":
        """Cast sparse features to single precision.

        Returns
        -------
        SparseTensor
            Tensor sharing coordinates and provenance with ``float32``
            features.
        """
        return self.replace_features(self.F.float())

    def __add__(self, other: Any) -> "SparseTensor":
        """Add another sparse tensor or a value to the feature matrix."""
        if isinstance(other, SparseTensor):
            if not len(self) and not len(other):
                return SparseTensor.empty_like(self)
            return self._wrap(self.backend_tensor + other.backend_tensor)
        return self.replace_features(self.F + other)

    def __mul__(self, other: Any) -> "SparseTensor":
        """Multiply by another sparse tensor or a feature-wise value."""
        if isinstance(other, SparseTensor):
            if not len(self) and not len(other):
                return SparseTensor.empty_like(self)
            return self._wrap(self.backend_tensor * other.backend_tensor)
        return self.replace_features(self.F * other)

    def __truediv__(self, other: Any) -> "SparseTensor":
        """Divide by another sparse tensor or a feature-wise value."""
        if isinstance(other, SparseTensor):
            if not len(self) and not len(other):
                return SparseTensor.empty_like(self)
            return self._wrap(self.backend_tensor / other.backend_tensor)
        return self.replace_features(self.F / other)
