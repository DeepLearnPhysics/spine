"""Self-describing event-level tensor products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spine.utils.conditional import torch

from ..larcv.meta import Meta
from .base import DataProduct, TensorSchema

__all__ = ["TensorData"]

ArrayLike = np.ndarray | torch.Tensor


@dataclass(init=False)
class TensorData(DataProduct):
    """Self-describing tensor-like payload for one event.

    Attributes
    ----------
    features : np.ndarray
        Feature matrix associated with the parsed tensor.
    coordinate_data : np.ndarray, optional
        Complete coordinate matrix, typically with shape ``(N, 3)``. Use
        :attr:`coords` or :meth:`coordinates` for semantic access.
    meta : Meta, optional
        Geometry metadata used to convert voxel indices into detector
        coordinates.
    index_shifts : np.ndarray, optional
        Shifts applied to index-bearing feature columns during batching.
    index_cols : np.ndarray, optional
        Feature columns that store indices.
    remove_duplicates : bool, default False
        If `True`, drop duplicate coordinates during collation.
    sum_cols : np.ndarray, optional
        Feature columns that should be summed when duplicates are merged.
    avg_cols : np.ndarray, optional
        Feature columns that should be averaged when duplicates are merged.
    prec_col : int, optional
        Feature column used to break duplicate-coordinate ties.
    precedence : np.ndarray, optional
        Precedence ordering used with ``prec_col``.
    feats_only : bool, default False
        If `True`, the payload is feature-only and has no associated
        coordinate tensor.
    overlay_reference : str, optional
        Product key whose duplicate-cleaning row selection should be applied
        to this tensor during overlay.
    schema : TensorSchema, optional
        Logical coordinate, feature and overlay description. When omitted, a
        deterministic schema is inferred from the coordinate width.
    """

    product_type = "tensor"

    features: ArrayLike
    _coords: ArrayLike | None = None
    meta: Meta | None = None
    index_shifts: np.ndarray | None = None
    schema: TensorSchema = TensorSchema()

    def __init__(
        self,
        features: ArrayLike,
        coords: ArrayLike | None = None,
        meta: Meta | None = None,
        index_shifts: np.ndarray | None = None,
        schema: TensorSchema | None = None,
        coordinate_groups: dict[str, tuple[int, ...]] | None = None,
        feature_fields: dict[str, tuple[int, ...]] | None = None,
        index_cols: np.ndarray | None = None,
        remove_duplicates: bool = False,
        sum_cols: np.ndarray | None = None,
        avg_cols: np.ndarray | None = None,
        prec_col: int | None = None,
        precedence: np.ndarray | None = None,
        feats_only: bool = False,
        overlay_reference: str | None = None,
    ) -> None:
        """Initialize an event tensor and its logical schema.

        Parameters
        ----------
        features : numpy.ndarray or torch.Tensor
            Event feature values. One-dimensional inputs represent a single
            scalar field; two-dimensional inputs represent one row per item.
        coords : numpy.ndarray or torch.Tensor, optional
            Coordinates aligned row-for-row with ``features``.
        meta : Meta, optional
            Image metadata associated with the coordinate system.
        index_shifts : numpy.ndarray, optional
            Parser-provided index offsets consumed during collation.
        schema : TensorSchema, optional
            Complete logical schema. It is mutually exclusive with the
            individual schema keyword arguments below.
        coordinate_groups : dict[str, tuple[int, ...]], optional
            Named groups within the coordinate matrix.
        feature_fields : dict[str, tuple[int, ...]], optional
            Named groups within the feature matrix.
        index_cols : numpy.ndarray, optional
            Feature-relative index columns shifted during overlay.
        remove_duplicates : bool, default False
            Merge rows with duplicate coordinates during overlay.
        sum_cols : numpy.ndarray, optional
            Feature-relative columns summed when duplicates are merged.
        avg_cols : numpy.ndarray, optional
            Feature-relative columns averaged when duplicates are merged.
        prec_col : int, optional
            Feature-relative precedence column for duplicate resolution.
        precedence : numpy.ndarray, optional
            Ordering applied to values in ``prec_col``.
        feats_only : bool, default False
            Explicitly mark the product as having no coordinates.
        overlay_reference : str, optional
            Row-aligned product whose overlay selection should be reused.

        Raises
        ------
        ValueError
            If a complete schema is combined with individual schema fields.
        """
        # A complete schema is authoritative; accepting partial overrides would
        # make serialized metadata depend on constructor call order.
        if schema is not None and any(
            value is not None
            for value in (
                coordinate_groups,
                feature_fields,
                index_cols,
                sum_cols,
                avg_cols,
                prec_col,
                precedence,
                overlay_reference,
            )
        ):
            raise ValueError("Do not combine `schema` with schema keyword fields.")
        if schema is None:
            # Infer a stable baseline, then overlay optional semantic names
            schema = TensorSchema.infer(
                coords,
                features=features,
                feats_only=feats_only,
                index_cols=self._tuple(index_cols),
                remove_duplicates=remove_duplicates,
                sum_cols=self._tuple(sum_cols),
                avg_cols=self._tuple(avg_cols),
                prec_col=prec_col,
                precedence=self._tuple(precedence),
                overlay_reference=overlay_reference,
            )
            if coordinate_groups is not None or feature_fields is not None:
                values = schema.to_dict()
                if coordinate_groups is not None:
                    values["coordinate_groups"] = coordinate_groups
                if feature_fields is not None:
                    values["feature_fields"] = feature_fields
                schema = TensorSchema.from_dict(values)

        # Store physical arrays separately from their immutable logical schema
        self.features = features
        self._coords = coords
        self.meta = meta
        self.index_shifts = index_shifts
        self.schema = schema

    @staticmethod
    def _tuple(value: np.ndarray | None) -> tuple[int, ...] | None:
        """Normalize optional array metadata to immutable tuples."""
        return None if value is None else tuple(np.asarray(value).tolist())

    @classmethod
    def metadata(cls, schema: TensorSchema | None = None) -> dict[str, Any]:
        """Return metadata sufficient to reconstruct the product.

        Parameters
        ----------
        schema : TensorSchema, optional
            Logical schema to include in the serialized metadata.

        Returns
        -------
        dict
            Product type and, when provided, a serialized schema.
        """
        metadata = super().metadata()
        if schema is not None:
            metadata["schema"] = schema.to_dict()

        return metadata

    @property
    def coordinate_groups(self) -> dict[str, tuple[int, ...]]:
        """Return the named coordinate-column groups."""
        return self.schema.coordinate_groups

    @property
    def coordinate_data(self) -> ArrayLike | None:
        """Return the complete coordinate matrix without disambiguation.

        This interface is intended for collation, augmentation and
        serialization code which must operate on every coordinate group.
        Ordinary consumers should prefer :attr:`coords` or
        :meth:`coordinates`.
        """
        return self._coords

    @coordinate_data.setter
    def coordinate_data(self, value: ArrayLike | None) -> None:
        """Replace the complete coordinate matrix."""
        self._coords = value

    @property
    def coords(self) -> ArrayLike | None:
        """Return the sole coordinate group, or `None` when absent.

        Raises
        ------
        ValueError
            If the product advertises multiple coordinate groups.
        """
        if self._coords is None:
            return None
        return self.coordinates()

    @coords.setter
    def coords(self, value: ArrayLike | None) -> None:
        """Replace coordinates for backward-compatible mutation paths."""
        self._coords = value

    def coordinates(self, name: str | None = None) -> ArrayLike:
        """Return one named coordinate group.

        Parameters
        ----------
        name : str, optional
            Group name. It may be omitted only when the schema defines one
            coordinate group.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Coordinate columns aligned with the feature rows.

        Raises
        ------
        ValueError
            If coordinates are absent or the omitted group is ambiguous.
        KeyError
            If ``name`` is not present in the schema.
        """
        if self._coords is None:
            raise ValueError("This tensor product has no coordinates.")

        # Resolve an omitted name only when the schema is unambiguous
        if name is None:
            if len(self.coordinate_groups) != 1:
                raise ValueError(
                    "Coordinate group is ambiguous; specify one of "
                    f"{tuple(self.coordinate_groups)}."
                )
            name = next(iter(self.coordinate_groups))
        if name not in self.coordinate_groups:
            raise KeyError(f"Unknown coordinate group `{name}`.")

        return self._coords[:, self.coordinate_groups[name]]

    @property
    def data(self) -> ArrayLike:
        """Return the canonical packed event array."""
        if self._coords is None:
            return self.features

        # Preserve the physical backend while packing coordinates and features
        if isinstance(self._coords, np.ndarray):
            return np.concatenate((self._coords, self.features), axis=1)

        return torch.cat((self._coords, self.features), dim=1)

    def __len__(self) -> int:
        """Return the number of rows in the event product."""
        return len(self.features)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the canonical packed representation."""
        return self.data.shape

    def __getitem__(self, index: Any) -> Any:
        """Index the canonical packed representation."""
        return self.data[index]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Expose the packed representation through NumPy's array protocol.

        PyTorch-backed products are detached and transferred to CPU. This
        compatibility interface is most natural for feature-only products;
        coordinate-bearing consumers should generally use named accessors.
        """
        data = self.data

        # Convert PyTorch storage explicitly before applying NumPy options
        if isinstance(data, np.ndarray):
            array = np.asarray(data, dtype=dtype)
        else:
            array = data.detach().cpu().numpy()
            if dtype is not None:
                array = array.astype(dtype, copy=False)

        return array.copy() if copy else array

    def feature(self, name: str) -> ArrayLike:
        """Return one named feature field.

        Parameters
        ----------
        name : str
            Field name defined by :attr:`schema`.

        Raises
        ------
        KeyError
            If the schema does not define ``name``.
        """
        if name not in self.schema.feature_fields:
            raise KeyError(f"Unknown feature field `{name}`.")

        return self.features[:, self.schema.feature_fields[name]]

    @property
    def values(self) -> ArrayLike:
        """Return the primary feature as a one-dimensional array.

        Raises
        ------
        ValueError
            If the product contains no feature columns.
        """
        if self.features.ndim == 1:
            return self.features

        if self.features.ndim != 2 or self.features.shape[1] == 0:
            raise ValueError("`values` requires at least one feature column.")

        # Feature zero is the conventional charge/value input. Auxiliary
        # features remain available through `features` and `feature(...)`.
        return self.features[:, 0]

    @property
    def index_cols(self) -> np.ndarray | None:
        """Return packed index-bearing columns."""
        return self._array(self.schema.index_cols)

    @property
    def remove_duplicates(self) -> bool:
        """Whether overlay merges duplicate coordinate rows."""
        return self.schema.remove_duplicates

    @property
    def sum_cols(self) -> np.ndarray | None:
        """Return packed columns summed during duplicate merging."""
        return self._array(self.schema.sum_cols)

    @property
    def avg_cols(self) -> np.ndarray | None:
        """Return packed columns averaged during duplicate merging."""
        return self._array(self.schema.avg_cols)

    @property
    def prec_col(self) -> int | None:
        """Return the packed precedence column."""
        return self.schema.prec_col

    @property
    def precedence(self) -> np.ndarray | None:
        """Return duplicate precedence ordering."""
        return self._array(self.schema.precedence)

    @property
    def feats_only(self) -> bool:
        """Whether this product intentionally contains features only."""
        return self.schema.feats_only

    @property
    def overlay_reference(self) -> str | None:
        """Return the row-aligned overlay reference key."""
        return self.schema.overlay_reference

    @staticmethod
    def _array(value: tuple[int, ...] | None) -> np.ndarray | None:
        """Expose immutable schema columns through the historical array API."""
        return None if value is None else np.asarray(value, dtype=np.int64)
