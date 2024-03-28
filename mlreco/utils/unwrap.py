"""Module with the classes/functions needed to unwrap batched data."""

import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from .data_structures import TensorBatch, IndexBatch, EdgeIndexBatch
from .geometry import Geometry


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
        self.num_volumes = self.geo.num_modules if self.geo else 1
        self.remove_batch_col = remove_batch_col

    def __call__(self, data_dict, result_dict):
        """Main unwrapping function.
        
        Loops over the data and result keys and applies the unwrapping rules.
        Returns the unwrapped versions of the two dictionaries.

        Parameters
        ----------
        data_dict : dict
            Dictionary of input data (key, batched input)
        result_dict : dict
            Dictionary of output of trainval.forward (key, batched output)

        Returns
        -------
        dict
            Dictionary of unwrapped input data (key, [batch_size])
        dict
            Dictionary of unwrapped output data (key, [batch_size])
        """
        data_unwrapped, result_unwrapped = {}, {}
        for key, value in data_dict.items():
            data_unwrapped[key] = self._unwrap(key, value)
        for key, value in result_dict.items():
            result_unwrapped[key] = self._unwrap(key, value)

        return data_unwrapped, result_unwrapped

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
        assert np.isscalar(data) or len(data), (
                "Batch has length 0, should not happen.")

        # Dispatch to the correct unwrapping scheme
        if (np.isscalar(data) or 
            (isinstance(data, list) and np.isscalar(data[0]))):
            # If there is a single scalar for the entire batch or a list
            # of scalars (one per entry), return as is
            return data

        elif isinstance(data, TensorBatch):
            # If the data is a tensor, split it between its constituents
            return self._unwrap_tensor(data)

        elif isinstance(data, list) and isinstance(data[0], TensorBatch):
            # If the data is a tensor list, split each between its constituents
            data_split = [self._unwrap_tensor(t) for t in data]
            tensor_lists = []
            for b in range(self.batch_size):
                tensor_lists.append([l[b] for l in data_split])

            return tensor_lists

        elif isinstance(data, (IndexBatch, EdgeIndexBatch)):
            # If the data is an index, split it between its constituents
            return self._unwrap_index(data)

        else:
            raise ValueError(
                    f"Type of {key} not unwrappable: {type(data)}")

    def _unwrap_tensor(self, data):
        """Unwrap a batch of tensors into its constituents.

        Parameters
        ----------
        data : TensorBatch
            Tensor batch product
        """
        # If there is one volume, trivial
        if self.num_volumes == 1:
            if not self.remove_batch_col or data.batch_col is None:
                return data.split()
            else:
                data_nobc = TensorBatch(
                        data.tensor[:, data.batch_col+1:], data.counts)
                return data_nobc.split()

        # Otherwise, must shift coordinates back
        tensors = []
        batch_size = data.batch_size//self.num_volumes
        for b in range(batch_size):
            for v in range(self.num_volumes):
                idx = b*self.num_volumes + v
                tensor = data[idx]
                if v > 0 and data.coord_cols is not None:
                    for cols in data.coord_cols:
                        tensor[:, cols] = self.geo.translate(
                                tensor[:, cols], 0, v)
                if self.remove_batch_col and data.batch_col is not None:
                    tensor = tensor[:, data.batch_col+1:]

                tensors.append(tensor)

        return tensors

    def _unwrap_index(self, data):
        """Unwrap an index list into its constituents.

        Parameters
        ----------
        data : IndexBatch
            Index batch product
        """
        # If there is only one volume, trivial
        if self.num_volumes == 1:
            return data.split()

        # If there is more than one volume, merge them together
        batch_size = data.batch_size // self.num_volumes
        indexes = []
        for b in range(batch_size):
            index_list = []
            for v in range(self.num_volumes):
                idx = b*self.num_volumes + v
                offset = self.offsets[idx] - self.offsets[b*self.num_volumes]
                index_list.append(offset + data[idx])
            
            indexes.append(np.concatenate(index_list))

        return indexes
