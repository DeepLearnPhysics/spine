"""Shared contracts and schemas for self-describing SPINE data products."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np

__all__ = ["DataProduct", "TensorSchema"]


class DataProduct:
    """Base contract for one event worth of a SPINE data product.

    Product classes describe their logical contents independently of the
    parser which produced them. I/O code may use :attr:`product_type` and
    :meth:`metadata` to collate, overlay, serialize and reconstruct products.
    """

    product_type: ClassVar[str]
    overlay_method: ClassVar[str | None] = None

    @classmethod
    def metadata(cls) -> dict[str, Any]:
        """Return serializable class-level product metadata."""
        return {"product_type": cls.product_type}


@dataclass(frozen=True)
class TensorSchema:
    """Logical description of a tensor-like event product.

    Column indexes are expressed relative to the separate coordinate and
    feature matrices stored by :class:`TensorData`. A batched representation
    may prepend a batch column without changing this schema.

    Parameters
    ----------
    coordinate_groups : dict[str, tuple[int, ...]]
        Named groups of coordinate columns. Products with start and end points
        therefore advertise two independent three-dimensional groups.
    feature_fields : dict[str, tuple[int, ...]]
        Optional named feature columns.
    index_cols : tuple[int, ...], optional
        Feature columns which contain indexes into another namespace.
    remove_duplicates : bool, default False
        Whether overlay should merge rows with duplicate coordinates.
    sum_cols : tuple[int, ...], optional
        Feature columns summed while merging duplicate rows.
    avg_cols : tuple[int, ...], optional
        Feature columns averaged while merging duplicate rows.
    prec_col : int, optional
        Feature column used to resolve duplicate-row precedence.
    precedence : tuple[int, ...], optional
        Ordering associated with :attr:`prec_col`.
    feats_only : bool, default False
        Whether the product intentionally has no coordinate matrix.
    overlay_reference : str, optional
        Name of another row-aligned product whose overlay selection is reused.
    """

    coordinate_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    feature_fields: dict[str, tuple[int, ...]] = field(default_factory=dict)
    index_cols: tuple[int, ...] | None = None
    remove_duplicates: bool = False
    sum_cols: tuple[int, ...] | None = None
    avg_cols: tuple[int, ...] | None = None
    prec_col: int | None = None
    precedence: tuple[int, ...] | None = None
    feats_only: bool = False
    overlay_reference: str | None = None

    @classmethod
    def infer(
        cls,
        coords: np.ndarray | None,
        *,
        features: Any | None = None,
        feats_only: bool = False,
        **kwargs: Any,
    ) -> "TensorSchema":
        """Build a schema with deterministic field and coordinate names.

        Parameters
        ----------
        coords : numpy.ndarray, optional
            Representative coordinate matrix used to infer its width.
        features : array-like, optional
            Representative feature matrix used to infer its width.
        feats_only : bool, default False
            Suppress coordinate inference for an explicitly feature-only
            product.
        **kwargs : dict
            Additional :class:`TensorSchema` fields.

        Returns
        -------
        TensorSchema
            Immutable inferred schema.
        """
        # Infer semantic coordinate groups from the coordinate width
        groups: dict[str, tuple[int, ...]] = {}
        if coords is not None and not feats_only:
            width = coords.shape[1] if coords.ndim > 1 else 1
            if width == 3:
                groups["points"] = (0, 1, 2)
            elif width % 3 == 0:
                for index in range(width // 3):
                    lower = 3 * index
                    groups[f"points_{index}"] = tuple(range(lower, lower + 3))
            else:
                groups["coordinates"] = tuple(range(width))

        # Infer a stable name for scalar or vector feature payloads
        fields: dict[str, tuple[int, ...]] = {}
        if features is not None:
            width = features.shape[1] if features.ndim > 1 else 1
            if width == 1:
                fields["value"] = (0,)
            elif width > 1:
                fields["features"] = tuple(range(width))

        # Combine inferred names with the explicitly supplied schema options
        return cls(
            coordinate_groups=groups,
            feature_fields=fields,
            feats_only=feats_only,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe schema representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, metadata: dict[str, Any]) -> "TensorSchema":
        """Reconstruct a schema from YAML/JSON-safe metadata.

        Parameters
        ----------
        metadata : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        TensorSchema
            Schema with column collections restored as tuples.
        """
        # Restore named column groups from serialized lists to immutable tuples
        values = dict(metadata)
        for key in (
            "coordinate_groups",
            "feature_fields",
        ):
            values[key] = {
                name: tuple(columns) for name, columns in values.get(key, {}).items()
            }

        # Restore all optional flat column collections in the same way
        for key in ("index_cols", "sum_cols", "avg_cols", "precedence"):
            if values.get(key) is not None:
                values[key] = tuple(values[key])

        return cls(**values)
