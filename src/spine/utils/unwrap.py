"""Module with the classes/functions needed to unwrap batched data."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from spine.data import EdgeIndexBatch, IndexBatch, Meta, ObjectList, TensorBatch
from spine.geo import GeoManager

from .globals import BATCH_COL

__all__ = ["Unwrapper"]


class Unwrapper:
    """Unwraps batched data to its constituent entries.

    Class used to break down the batched input and output dictionaries into
    individual events. When passed through the model, the input is concatenated
    into single tensors/arrays for faster processing; this class breaks the
    output down event-wise to be human-readable.
    """

    def __init__(self, meta_key: str = "meta", remove_batch_col: bool = False):
        """Initialize the unwrapper.

        Parameters
        ----------
        remove_batch_col : bool
             Remove column which specifies batch ID from the unwrapped tensors
        """
        # Fetch the geometry instance, if available
        self.geo = GeoManager.get_instance_if_initialized()

        # Store parameters
        self.num_volumes = self.geo.tpc.num_modules if self.geo else 1
        self.meta_key = meta_key
        self.remove_batch_col = remove_batch_col
        self.batch_size = None

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main unwrapping function.

        Loops over the data keys and applies the unwrapping rules. Returns the
        unwrapped versions of the dictionary

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data products

        Returns
        -------
        Dict[str, Any]
            Dictionary of unwrapped data products
        """
        # If there are multiple volumes, fetch the metadata
        meta = None
        if self.num_volumes > 1 and self.meta_key in data:
            meta = data[self.meta_key]

        # Loop over data products and unwrap them
        data_unwrapped = {}
        self.batch_size = len(data["index"])
        for key, value in data.items():
            data_unwrapped[key] = self._unwrap(key, value, meta)

        return data_unwrapped

    def _unwrap(self, key: str, data: Any, meta: Optional[List[Meta]] = None) -> Any:
        """Routes set of data to the appropriate unwrapping scheme.

        Parameters
        ----------
        key : str
            Name of the data product to unwrap
        data : Any
            Data product
        meta : List[Meta], optional
            Metadata associated with each image in the batch

        Returns
        -------
        Any
            Unwrapped data product
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
            return self._unwrap_tensor(data, meta)

        elif isinstance(data, list) and isinstance(data[0], TensorBatch):
            # If the data is a tensor list, split each between its constituents
            data_split = [self._unwrap_tensor(t, meta) for t in data]
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

    def _unwrap_tensor(
        self, data: TensorBatch, meta: Optional[List[Meta]] = None
    ) -> List[np.ndarray]:
        """Unwrap a batch of tensors into its constituents.

        Parameters
        ----------
        data : TensorBatch
            Tensor batch product
        meta : Meta, optional
            Metadata associated with each image in the batch

        Returns
        -------
        List[np.ndarray]
            List of unwrapped tensors
        """
        # If there is one volume per batch, trivial
        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            if not self.remove_batch_col or not data.has_batch_col:
                return data.split()
            else:
                data_nobc = TensorBatch(data.tensor[:, BATCH_COL + 1 :], data.counts)
                return data_nobc.split()

        # Otherwise, must shift coordinates back
        assert (
            self.geo is not None
        ), "Geometry must be initialized to unwrap tensors from multiple volumes."
        assert (
            meta is not None and len(meta) == self.batch_size
        ), "Metadata must be provided to unwrap tensors from multiple volumes."

        tensors = []
        batch_size = data.batch_size // self.num_volumes
        for b in range(batch_size):
            tensor_list = []
            for v in range(self.num_volumes):
                idx = b * self.num_volumes + v
                tensor = data[idx]
                if v > 0 and data.coord_cols is not None:
                    for cols in data.coord_cols.reshape(-1, 3):
                        tensor[:, cols] = self.geo.translate(
                            tensor[:, cols], 0, v, 1.0 / meta[b].size
                        )
                if self.remove_batch_col and data.has_batch_col:
                    tensor = tensor[:, BATCH_COL + 1 :]

                tensor_list.append(tensor)

            tensors.append(np.concatenate(tensor_list))

        return tensors

    def _unwrap_index(
        self, data: Union[IndexBatch, EdgeIndexBatch]
    ) -> Union[List[np.ndarray], List[ObjectList]]:
        """Unwrap an index list into its constituents.

        Parameters
        ----------
        data : Union[IndexBatch, EdgeIndexBatch]
            Index batch product

        Returns
        -------
        Union[List[np.ndarray], List[ObjectList]]
            List of unwrapped indexes
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

        # Cast the index lists to ObjectList, in case they are empty
        if isinstance(data, IndexBatch) and data.is_list:
            shape = (0, data.shape[1]) if len(data.shape) == 2 else 0
            default = np.empty(shape, dtype=np.int64)
            indexes_obl = []
            for index in indexes:
                indexes_obl.append(ObjectList(index, default=default))

            return indexes_obl

        return indexes
