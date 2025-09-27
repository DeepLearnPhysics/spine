"""Module with the classes/functions needed to unwrap batched data."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from spine.data import EdgeIndexBatch, IndexBatch, ObjectList, TensorBatch

from .geo import Geometry
from .globals import BATCH_COL

__all__ = ["Unwrapper"]


class Unwrapper:
    """Unwraps batched data to its constituent entries.

    Class used to break down the batched input and output dictionaries into
    individual events. When passed through the model, the input is concatenated
    into single tensors/arrays for faster processing; this class breaks the
    output down event-wise to be human-readable.
    """

    def __init__(self, geometry=None, remove_batch_col=False):
        """Initialize the unwrapper.

        Parameters
        ----------
        geometry : Geometry
             Detector geometry (needed if the input was split in
             different volumes)
        remove_batch_col : bool
             Remove column which specifies batch ID from the unwrapped tensors
        """
        self.geo = geometry
        self.num_volumes = self.geo.tpc.num_modules if self.geo else 1
        self.remove_batch_col = remove_batch_col
        self.batch_size = None

    def __call__(self, data):
        """Main unwrapping function.

        Loops over the data keys and applies the unwrapping rules. Returns the
        unwrapped versions of the dictionary

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        dict
            Dictionary of unwrapped data products
        """
        data_unwrapped = {}
        self.batch_size = len(data["index"])
        for key, value in data.items():
            data_unwrapped[key] = self._unwrap(key, value)

        return data_unwrapped

    def _unwrap(self, key, data):
        """Routes set of data to the appropriate unwrapping scheme.

        Parameters
        ----------
        key : str
            Name of the data product to unwrap
        data : list
            Data product
        """
        # Data should never be an empty list
        dim = len(getattr(data, "shape", [0]))
        assert (
            np.isscalar(data) or dim == 0 or len(data)
        ), "Batch has length 0, should not happen."

        # Dispatch to the correct unwrapping scheme
        if (
            np.isscalar(data)
            or dim == 0
            or (isinstance(data, list) and not isinstance(data[0], TensorBatch))
        ):
            # If there is a single scalar for the entire batch or a simple list
            # of objects (one per entry), return as is
            return data

        elif isinstance(data, TensorBatch):
            # If the data is a tensor, split it between its constituents
            return self._unwrap_tensor(data)

        elif isinstance(data, list) and isinstance(data[0], TensorBatch):
            # If the data is a tensor list, split each between its constituents
            data_split = [self._unwrap_tensor(t) for t in data]
            tensor_lists = []
            batch_size = data[0].batch_size // self.num_volumes
            for b in range(batch_size):
                tensor_lists.append([l[b] for l in data_split])

            return tensor_lists

        elif isinstance(data, (IndexBatch, EdgeIndexBatch)):
            # If the data is an index, split it between its constituents
            return self._unwrap_index(data)

        else:
            raise ValueError(f"Type of {key} not unwrappable: {type(data)}")

    def _unwrap_tensor(self, data):
        """Unwrap a batch of tensors into its constituents.

        Parameters
        ----------
        data : TensorBatch
            Tensor batch product
        """
        # If there is one volume, trivial
        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            if not self.remove_batch_col or not data.has_batch_col:
                return data.split()
            else:
                data_nobc = TensorBatch(data.tensor[:, BATCH_COL + 1 :], data.counts)
                return data_nobc.split()

        # Otherwise, must shift coordinates back
        tensors = []
        batch_size = data.batch_size // self.num_volumes
        for b in range(batch_size):
            tensor_list = []
            for v in range(self.num_volumes):
                idx = b * self.num_volumes + v
                tensor = data[idx]
                if v > 0 and data.coord_cols is not None:
                    for cols in data.coord_cols.reshape(-1, 3):
                        # TODO: Hacky as hell, fix it
                        tensor[:, cols] = self.geo.translate(
                            tensor[:, cols], 0, v, 1.0 / 0.3
                        )
                if self.remove_batch_col and data.has_batch_col:
                    tensor = tensor[:, BATCH_COL + 1 :]

                tensor_list.append(tensor)

            tensors.append(np.concatenate(tensor_list))

        return tensors

    def _unwrap_index(self, data):
        """Unwrap an index list into its constituents.

        Parameters
        ----------
        data : IndexBatch
            Index batch product
        """
        # Unwrap
        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            # If there is only one volume, trivial
            indexes = data.split()

        else:
            # If there is more than one volume, merge them together
            batch_size = data.batch_size // self.num_volumes
            indexes = []
            for b in range(batch_size):
                index_list = []
                for v in range(self.num_volumes):
                    idx = b * self.num_volumes + v
                    offset = data.offsets[idx] - data.offsets[b * self.num_volumes]
                    index_list.append(offset + data[idx])

                indexes.append(np.concatenate(index_list))

        # Cast the indexes to ObjectList, in case they are empty
        if isinstance(data, IndexBatch) and data.is_list:
            shape = (0, data.shape[1]) if len(data.shape) == 2 else 0
            default = np.empty(shape, dtype=np.int64)
            for i, index in enumerate(indexes):
                indexes[i] = ObjectList(index, default=default)

        return indexes
