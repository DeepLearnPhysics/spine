"""Module that contains data structures which hold standard parser outputs."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from spine.constants import VALUE_COL
from spine.data import Meta, ObjectList

__all__ = ["ParserTensor", "ParserObjectList"]


@dataclass
class ParserTensor:
    """Class which holds all the elements necessary to build a sparse tensor.

    Attributes
    ----------
    features : np.ndarray
        (N, N_f) Feature vectors
    coords : np.ndarray, optional
        (N, 3) Sparse tensor coordinates (voxel indexes)
    meta : Meta, optional
        Metadata to convert from voxel ID to detector coordinates
    global_shift : int, optional
        Global shift to apply to all features to prevent overlap
    index_shifts : np.ndarray, optional
        (C) Shifts to apply to index columns to prevent overlap
    index_cols : np.ndarray, optional
        (C) Columns which contain indexes
    remove_duplicates : bool, default False
        If `True`, remove duplicated voxel coordinates
    sum_cols : np.ndarray, optional
        (S) Columns which should be summed when removing duplicates
    prec_col : int, optional
        Column to be used as a precedence source when removing duplicates
    precedence : np.ndarray, optional
        Order of precedence among the classes in prec_col
    feats_only : np.ndarray, default False
        If `True`, only the features of the sparse tensor are exposed
    """

    features: np.ndarray
    coords: np.ndarray | None = None
    meta: Meta | None = None
    global_shift: int | None = None
    index_shifts: np.ndarray | None = None
    index_cols: np.ndarray | None = None
    remove_duplicates: bool = False
    sum_cols: np.ndarray | None = None
    prec_col: int | None = None
    precedence: np.ndarray | None = None
    feats_only: bool = False

    @property
    def feat_index_cols(self) -> np.ndarray | None:
        """Returns the index columns for the feature tensor.

        Returns
        -------
        np.ndarray
            Index columns in the feature tensor
        """
        if self.index_cols is None:
            return self.index_cols

        return self.index_cols - VALUE_COL

    @property
    def feat_sum_cols(self) -> np.ndarray | None:
        """Returns the columns to be summed in the feature tensor.

        Returns
        -------
        np.ndarray
            Columns to be summed in the feature tensor
        """
        if self.sum_cols is None:
            return self.sum_cols

        return self.sum_cols - VALUE_COL

    @property
    def feat_prec_col(self) -> int | None:
        """Returns the column providing a precedence source in the feature
        tensor.

        Returns
        -------
        int
            Column providing a precedence source in the feature tensor
        """
        if self.prec_col is None or self.prec_col < 0:
            return self.prec_col

        return self.prec_col - VALUE_COL


class ParserObjectList(ObjectList):
    """Object list with index shifting instructions.

    Attributes
    ----------
    index_shifts : Union[int, Dict[str, int]]
        Shift(s) to apply to the index attribute of the objects in the list
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
        object_list : List[object]
            Object list
        default : object
            Default object class to use to type the list, if it is empty
        index_shifts : Union[int, Dict[str, int]], optional
            Shift(s) to apply to the index attribute of the objects in the list
        """
        # Initialize the underlying object list
        super().__init__(object_list, default)

        # Store the index shifts
        if index_shifts is not None:
            self.index_shifts = index_shifts
        else:
            self.index_shifts = len(object_list)

    @property
    def to_object_list(self):
        """Cast to the underlying ObjectList (drop index shifts).

        Returns
        -------
        ObjectList
            Underlying object list
        """
        return ObjectList(self, default=self.default)
