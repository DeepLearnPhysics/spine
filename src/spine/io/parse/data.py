"""Data structures used as canonical outputs of IO parsers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spine.constants import VALUE_COL
from spine.data import Meta, ObjectList

__all__ = ["ParserTensor", "ParserObjectList"]


@dataclass
class ParserTensor:
    """Container describing a parsed sparse tensor or index-style payload.

    Attributes
    ----------
    features : np.ndarray or list[np.ndarray]
        Feature matrix or jagged list of index arrays.
    coords : np.ndarray, optional
        Sparse tensor coordinates, typically with shape ``(N, 3)``.
    meta : Meta, optional
        Geometry metadata used to convert voxel indices into detector
        coordinates.
    global_shift : int, optional
        Global index shift used when batching graph-style payloads.
    single_counts : np.ndarray, optional
        Per-element sizes for jagged index payloads.
    index_shifts : np.ndarray, optional
        Shifts applied to index-bearing feature columns during batching.
    index_cols : np.ndarray, optional
        Feature columns that store indices.
    remove_duplicates : bool, default False
        If `True`, drop duplicate coordinates during collation.
    sum_cols : np.ndarray, optional
        Feature columns that should be summed when duplicates are merged.
    prec_col : int, optional
        Feature column used to break duplicate-coordinate ties.
    precedence : np.ndarray, optional
        Precedence ordering used with ``prec_col``.
    feats_only : bool, default False
        If `True`, the payload is feature-only and has no associated
        coordinate tensor.
    """

    features: np.ndarray | list[np.ndarray]
    coords: np.ndarray | None = None
    meta: Meta | None = None
    global_shift: int | None = None
    single_counts: np.ndarray | None = None
    index_shifts: np.ndarray | None = None
    index_cols: np.ndarray | None = None
    remove_duplicates: bool = False
    sum_cols: np.ndarray | None = None
    prec_col: int | None = None
    precedence: np.ndarray | None = None
    feats_only: bool = False

    @property
    def feat_index_cols(self) -> np.ndarray | None:
        """Return index-bearing columns expressed in feature-only coordinates.

        Returns
        -------
        np.ndarray, optional
            Feature-column indices corresponding to :attr:`index_cols`.
        """
        if self.index_cols is None:
            return self.index_cols

        return self.index_cols - VALUE_COL

    @property
    def feat_sum_cols(self) -> np.ndarray | None:
        """Return duplicate-summed columns in feature-only coordinates.

        Returns
        -------
        np.ndarray, optional
            Feature-column indices corresponding to :attr:`sum_cols`.
        """
        if self.sum_cols is None:
            return self.sum_cols

        return self.sum_cols - VALUE_COL

    @property
    def feat_prec_col(self) -> int | None:
        """Return the precedence column in feature-only coordinates.

        Returns
        -------
        int, optional
            Feature-column index corresponding to :attr:`prec_col`.
        """
        if self.prec_col is None or self.prec_col < 0:
            return self.prec_col

        return self.prec_col - VALUE_COL


class ParserObjectList(ObjectList):
    """Object list with index shifting instructions.

    Attributes
    ----------
    index_shifts : int or dict[str, int]
        Shift(s) to apply to object index attributes during collation.
    """

    def __init__(
        self,
        object_list: list[object],
        default: Any,
        index_shifts: int | dict[str, int] | None = None,
    ) -> None:
        """Initialize the list and the default value.

        Parameters
        ----------
        object_list : list[object]
            Parsed objects associated with one event entry.
        default : object
            Default object used to type an empty list.
        index_shifts : int or dict[str, int], optional
            Shift(s) to apply to object index attributes during batching.
        """
        # Initialize the underlying object list
        super().__init__(object_list, default)

        # Store the index shifts
        if index_shifts is not None:
            self.index_shifts = index_shifts
        else:
            self.index_shifts = len(object_list)

    @property
    def to_object_list(self) -> ObjectList:
        """Drop parser-specific batching metadata and return a plain ObjectList.

        Returns
        -------
        ObjectList
            Underlying object list without ``index_shifts`` metadata.
        """
        return ObjectList(self, default=self.default)
