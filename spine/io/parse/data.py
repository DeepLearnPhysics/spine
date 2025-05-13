"""Module that contains data structures which hold standard parser outputs."""

from dataclasses import dataclass

import numpy as np

from spine.data import Meta, ObjectList

from spine.utils.globals import VALUE_COL

__all__ = ['ParserTensor', 'ParserObjectList']


@dataclass
class ParserTensor:
    """Class which holds all the elements necessary to build a sparse tensor.

    Attributes
    ----------
    features : np.ndarray
        (N, N_f) Feature vectors
    coords : np.ndarray, optional
        (N, 3) Voxel coordinates
    meta : Meta, optional
        Metadata to convert from voxel ID to detector coordinates
    index_shifts : np.ndarray, optional
        (C/R) Shifts to apply to index columns to prevent overlap
    index_cols : np.ndarray, optional
        (C) Columns which contain indexes
    index_rows : np.ndarray, optional
        (R) Rows which contain indexes
    """
    features: np.ndarray
    coords: np.ndarray = None
    meta: Meta = None
    index_shifts: np.ndarray = None
    index_cols: np.ndarray = None
    index_rows: np.ndarray = None

    @property
    def feat_index_cols(self):
        """Returns the index columns for the feature tensor.

        Returns
        -------
        np.ndarray
            Index columns in the feature tensor
        """
        if self.index_cols is None or self.coords is None:
            return self.index_cols

        return self.index_cols - VALUE_COL


class ParserObjectList(ObjectList):
    """Object list with index shifting instructions.

    Attributes
    ----------
    index_shifts : Union[int, Dict[str, int]]
        Shift(s) to apply to the index attribute of the objects in the list
    """

    def __init__(self, object_list, default, index_shifts=None):
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
        return ObjectList(self)
