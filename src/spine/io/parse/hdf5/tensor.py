"""Lightweight parsers for cached HDF5 tensor products."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.constants import COORD_COLS_LO, VALUE_COL
from spine.data import Meta

from ..base import ParserBase
from ..data import ParserTensor

__all__ = [
    "HDF5TensorParser",
    "HDF5ClusterTensorParser",
    "HDF5FeatureTensorParser",
]


class HDF5TensorParser(ParserBase):
    """Build a sparse-tensor :class:`ParserTensor` from a cached HDF5 tensor."""

    name = "tensor"
    returns = "tensor"

    def __init__(
        self,
        dtype: str,
        has_batch_col: bool = True,
        coord_start_col: int = COORD_COLS_LO,
        feature_start_col: int = VALUE_COL,
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
            returned :class:`ParserTensor`.
        feature_cols : sequence[int], optional
            Optional feature-column indices to keep after splitting coordinates
            and features.
        **kwargs : dict, optional
            Parser configuration forwarded to :class:`ParserBase`.
        """
        super().__init__(dtype, meta_event=meta_event, **kwargs)
        self.has_batch_col = has_batch_col
        self.coord_start_col = coord_start_col
        self.feature_start_col = feature_start_col
        self.feature_cols = None
        if feature_cols is not None:
            self.feature_cols = np.asarray(feature_cols, dtype=np.int64)

    def __call__(self, trees: dict[str, Any]) -> ParserTensor:
        """Parse one cached entry into a sparse-tensor parser payload."""
        return self.process(**self.get_input_data(trees))

    def process(
        self, tensor_event: np.ndarray, meta_event: Meta | None = None
    ) -> ParserTensor:
        """Split one cached tensor into coordinates, features, and metadata."""
        tensor = np.asarray(tensor_event, dtype=self.ftype)
        if tensor.ndim != 2:
            raise ValueError(
                "Cached sparse tensors must be 2D. "
                f"Received an array with shape {tensor.shape}."
            )

        coords = tensor[:, self.coord_start_col : self.feature_start_col].astype(
            self.itype
        )
        features = tensor[:, self.feature_start_col :]
        if self.feature_cols is not None:
            features = features[:, self.feature_cols]

        if self.has_batch_col and self.coord_start_col < 1:
            raise ValueError(
                "`coord_start_col` must be at least 1 when `has_batch_col=True`."
            )

        return ParserTensor(coords=coords, features=features, meta=meta_event)


class HDF5ClusterTensorParser(HDF5TensorParser):
    """Build a cluster-label :class:`ParserTensor` from cached HDF5 tensors."""

    name = "cluster_tensor"

    def __init__(
        self,
        dtype: str,
        index_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
        sum_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
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
        super().__init__(dtype, **kwargs)
        self.index_cols = None if index_cols is None else np.asarray(index_cols)
        self.sum_cols = None if sum_cols is None else np.asarray(sum_cols)
        self.prec_col = prec_col
        self.precedence = None if precedence is None else np.asarray(precedence)
        self.remove_duplicates = remove_duplicates

    def process(
        self, tensor_event: np.ndarray, meta_event: Meta | None = None
    ) -> ParserTensor:
        """Split one cached cluster tensor and restore cluster parser semantics."""
        tensor = super().process(tensor_event=tensor_event, meta_event=meta_event)
        tensor.index_cols = self.index_cols
        tensor.sum_cols = self.sum_cols
        tensor.prec_col = self.prec_col
        tensor.precedence = self.precedence
        tensor.remove_duplicates = self.remove_duplicates
        return tensor


class HDF5FeatureTensorParser(ParserBase):
    """Build a feature-only :class:`ParserTensor` from a cached HDF5 array."""

    name = "feature_tensor"
    returns = "tensor"

    def __init__(
        self,
        dtype: str,
        feature_cols: list[int] | tuple[int, ...] | np.ndarray | None = None,
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
        **kwargs : dict, optional
            Parser configuration forwarded to :class:`ParserBase`.
        """
        super().__init__(dtype, **kwargs)
        self.feature_cols = None
        if feature_cols is not None:
            self.feature_cols = np.asarray(feature_cols, dtype=np.int64)

    def __call__(self, trees: dict[str, Any]) -> ParserTensor:
        """Parse one cached entry into a feature-only parser tensor.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        ParserTensor
            Feature-only parser tensor built from the cached array.
        """
        return self.process(**self.get_input_data(trees))

    def process(self, tensor_event: np.ndarray) -> ParserTensor:
        """Cast one cached per-entry array into a feature-only parser tensor.

        Parameters
        ----------
        tensor_event : np.ndarray
            Cached feature array for one event entry.

        Returns
        -------
        ParserTensor
            Feature-only parser tensor with ``features`` cast to the parser
            float dtype.
        """
        features = np.asarray(tensor_event, dtype=self.ftype)
        if self.feature_cols is not None:
            if features.ndim != 2:
                raise ValueError(
                    "Feature ablation requires a 2D cached feature tensor. "
                    f"Received an array with shape {features.shape}."
                )
            features = features[:, self.feature_cols]

        return ParserTensor(features=features, feats_only=True)
