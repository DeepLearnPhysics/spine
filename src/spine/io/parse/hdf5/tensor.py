"""Lightweight parsers for cached HDF5 tensor products."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import Meta, TensorData

from ..base import ParserBase

__all__ = [
    "HDF5TensorParser",
    "HDF5ClusterTensorParser",
    "HDF5FeatureTensorParser",
]


class HDF5TensorParser(ParserBase):
    """Build a sparse-tensor :class:`TensorData` from a cached HDF5 tensor."""

    name = "tensor"

    def __init__(
        self,
        dtype: str,
        has_batch_col: bool = True,
        coord_start_col: int = 1,
        feature_start_col: int = 4,
        meta_event: str | None = None,
        feature_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the cached tensor parser.

        Parameters
        ----------
        dtype : str
            Floating-point dtype used by parser outputs.
        has_batch_col : bool, default True
            If `True`, the cached tensor is assumed to store a leading batch-id
            column before the coordinates.
        coord_start_col : int, default 1
            Column index at which the coordinate block starts.
        feature_start_col : int, default 4
            Column index at which the feature block starts.
        meta_event : str, optional
            HDF5 product name that stores the metadata object to inject into the
            returned :class:`TensorData`.
        feature_cols : sequence[int], optional
            Optional feature-column indices to keep after splitting coordinates
            and features.
        **kwargs : dict, optional
            Parser configuration forwarded to :class:`ParserBase`.
        """
        # Register physical inputs before retaining column-layout options
        super().__init__(dtype, meta_event=meta_event, **kwargs)
        self.has_batch_col = has_batch_col
        self.coord_start_col = coord_start_col
        self.feature_start_col = feature_start_col
        self.feature_cols = None

        # Normalize optional feature ablation to stable integer columns
        if feature_cols is not None:
            self.feature_cols = np.asarray(feature_cols, dtype=np.int64)

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Parse one cached entry into a sparse-tensor product.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        TensorData
            Sparse event tensor containing coordinates, features, and optional
            spatial metadata.
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self, tensor_event: np.ndarray, meta_event: Meta | None = None
    ) -> TensorData:
        """Split one cached tensor into coordinates, features, and metadata.

        Parameters
        ----------
        tensor_event : np.ndarray
            Cached two-dimensional sparse tensor.
        meta_event : Meta, optional
            Spatial metadata associated with the cached tensor.

        Returns
        -------
        TensorData
            Self-describing sparse event tensor.

        Raises
        ------
        ValueError
            If the physical tensor is not two-dimensional or its configured
            coordinate columns cannot follow a retained batch column.
        """
        # Normalize and validate the cached physical matrix
        tensor = np.asarray(tensor_event, dtype=self.ftype)
        if tensor.ndim != 2:
            raise ValueError(
                "Cached sparse tensors must be 2D. "
                f"Received an array with shape {tensor.shape}."
            )

        # Split coordinates from features using the configured storage layout
        coords = tensor[:, self.coord_start_col : self.feature_start_col].astype(
            self.itype
        )
        features = tensor[:, self.feature_start_col :]
        if self.feature_cols is not None:
            features = features[:, self.feature_cols]

        # A retained batch column must precede the event coordinate block
        if self.has_batch_col and self.coord_start_col < 1:
            raise ValueError(
                "`coord_start_col` must be at least 1 when `has_batch_col=True`."
            )

        return TensorData(coords=coords, features=features, meta=meta_event)


class HDF5ClusterTensorParser(HDF5TensorParser):
    """Build a cluster-label :class:`TensorData` from cached HDF5 tensors."""

    name = "cluster_tensor"

    def __init__(
        self,
        dtype: str,
        index_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
        sum_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
        avg_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
        prec_col: int | None = None,
        precedence: list[int] | tuple[int, ...] | np.ndarray | None = None,
        remove_duplicates: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the cached cluster-tensor parser.

        Parameters
        ----------
        dtype : str
            Floating-point dtype used by parser outputs.
        index_cols : sequence[int], optional
            Feature columns that carry indices and should be shifted when
            collating batches.
        sum_cols : sequence[int], optional
            Feature columns that should be summed when duplicate coordinates are
            merged.
        avg_cols : sequence[int], optional
            Feature columns that should be averaged when duplicate coordinates
            are merged.
        prec_col : int, optional
            Feature column used to resolve duplicate-coordinate precedence.
        precedence : sequence[int], optional
            Ordering applied to ``prec_col`` when duplicate coordinates are
            merged.
        remove_duplicates : bool, default True
            If `True`, mark the returned parser tensor for duplicate removal.
        **kwargs : dict, optional
            Tensor-parser configuration forwarded to :class:`HDF5TensorParser`.
        """
        # Initialize the physical tensor parser, then restore overlay semantics
        super().__init__(dtype, **kwargs)
        self.index_cols = None if index_cols is None else np.asarray(index_cols)
        self.sum_cols = None if sum_cols is None else np.asarray(sum_cols)
        self.avg_cols = None if avg_cols is None else np.asarray(avg_cols)
        self.prec_col = prec_col
        self.precedence = None if precedence is None else np.asarray(precedence)
        self.remove_duplicates = remove_duplicates

    def process(
        self, tensor_event: np.ndarray, meta_event: Meta | None = None
    ) -> TensorData:
        """Split a cached cluster tensor and restore overlay semantics.

        Parameters
        ----------
        tensor_event : np.ndarray
            Cached two-dimensional cluster tensor.
        meta_event : Meta, optional
            Spatial metadata associated with the cached tensor.

        Returns
        -------
        TensorData
            Sparse event tensor carrying cluster-specific index shifting and
            duplicate-reduction rules.
        """
        # Reuse physical splitting before attaching cluster-specific metadata
        tensor = super().process(tensor_event=tensor_event, meta_event=meta_event)

        return TensorData(
            coords=tensor.coordinate_data,
            features=tensor.features,
            meta=tensor.meta,
            index_cols=self.index_cols,
            sum_cols=self.sum_cols,
            avg_cols=self.avg_cols,
            prec_col=self.prec_col,
            precedence=self.precedence,
            remove_duplicates=self.remove_duplicates,
        )


class HDF5FeatureTensorParser(ParserBase):
    """Build a feature-only :class:`TensorData` from a cached HDF5 array."""

    name = "feature_tensor"

    def __init__(
        self,
        dtype: str,
        feature_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
        remove_duplicates: bool = False,
        overlay_reference: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the cached feature-tensor parser.

        Parameters
        ----------
        dtype : str
            Floating-point dtype used by parser outputs.
        feature_cols : sequence[int], optional
            Optional list of feature-column indices to keep from the cached
            tensor. When provided, this acts as a feature ablation step before
            the parser tensor is returned.
        remove_duplicates : bool, default False
            If `True`, require an ``overlay_reference`` when overlaying this
            feature-only tensor.
        overlay_reference : str, optional
            Product key whose duplicate-cleaning row selection should be used
            for this tensor when overlaying.
        **kwargs : dict, optional
            Parser configuration forwarded to :class:`ParserBase`.
        """
        # Register the cached tensor and normalize optional feature selection
        super().__init__(dtype, **kwargs)
        self.feature_cols = None
        if feature_cols is not None:
            self.feature_cols = np.asarray(feature_cols, dtype=np.int64)
        self.remove_duplicates = remove_duplicates
        self.overlay_reference = overlay_reference

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Parse one cached entry into a feature-only parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        TensorData
            Feature-only parser tensor built from the cached array.
        """
        return self.process(**self.get_input_data(trees))

    def process(self, tensor_event: np.ndarray) -> TensorData:
        """Cast one cached per-entry array into a feature-only parser tensor.

        Parameters
        ----------
        tensor_event : np.ndarray
            Cached feature array for one event entry.

        Returns
        -------
        TensorData
            Feature-only parser tensor with ``features`` cast to the parser
            float dtype.
        """
        # Cast the complete cached payload before applying optional ablation
        features = np.asarray(tensor_event, dtype=self.ftype)
        if self.feature_cols is not None:
            if features.ndim != 2:
                raise ValueError(
                    "Feature ablation requires a 2D cached feature tensor. "
                    f"Received an array with shape {features.shape}."
                )
            features = features[:, self.feature_cols]

        # Mark the result explicitly as coordinate-free for collation/overlay
        return TensorData(
            features=features,
            remove_duplicates=self.remove_duplicates,
            feats_only=True,
            overlay_reference=self.overlay_reference,
        )
