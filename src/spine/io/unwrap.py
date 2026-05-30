"""Module with the classes/functions needed to unwrap batched data."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from spine.constants import BATCH_COL
from spine.data import EdgeIndexBatch, IndexBatch, Meta, ObjectList, TensorBatch
from spine.geo import GeoManager

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
        self.geo = GeoManager.get_instance_if_initialized()

        self.num_volumes = self.geo.tpc.num_modules if self.geo else 1
        self.meta_key = meta_key
        self.remove_batch_col = remove_batch_col
        self.batch_size = None

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main unwrapping function."""
        meta = None
        if self.num_volumes > 1 and self.meta_key in data:
            meta = data[self.meta_key]

        data_unwrapped = {}
        self.batch_size = len(data["index"])
        for key, value in data.items():
            data_unwrapped[key] = self._unwrap(key, value, meta)
            if isinstance(value, (IndexBatch, EdgeIndexBatch)):
                span_key = f"{key}_spans"
                if span_key not in data:
                    data_unwrapped[span_key] = self._unwrap_index_spans(value)

        return data_unwrapped

    def _unwrap(self, key: str, data: Any, meta: Optional[List[Meta]] = None) -> Any:
        """Route data to the appropriate unwrapping scheme."""
        if isinstance(data, (list, tuple)) and len(data) == 0:
            raise ValueError(f"Batched data for {key} is an empty list, cannot unwrap.")
        if self.batch_size is None:
            raise ValueError("Batch size should be set before unwrapping.")

        dim = len(getattr(data, "shape", (0,)))
        if (
            np.isscalar(data)
            or dim == 0
            or (isinstance(data, list) and not isinstance(data[0], TensorBatch))
        ):
            return data

        if isinstance(data, TensorBatch):
            return self._unwrap_tensor(data, meta)

        if isinstance(data, list) and isinstance(data[0], TensorBatch):
            data_split = [self._unwrap_tensor(t, meta) for t in data]
            tensor_lists = []
            for batch_id in range(self.batch_size):
                tensor_lists.append([value[batch_id] for value in data_split])

            return tensor_lists

        if isinstance(data, (IndexBatch, EdgeIndexBatch)):
            return self._unwrap_index(data)

        raise ValueError(f"Type of {key} not unwrappable: {type(data)}")

    def _unwrap_tensor(
        self, data: TensorBatch, meta: Optional[List[Meta]] = None
    ) -> List[np.ndarray]:
        """Unwrap a batch of tensors into its constituents."""
        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            if not self.remove_batch_col or not data.has_batch_col:
                return data.split()

            data_nobc = TensorBatch(data.tensor[:, BATCH_COL + 1 :], data.counts)
            return data_nobc.split()

        if self.geo is None:
            raise ValueError(
                "Geometry must be initialized to unwrap tensors from multiple volumes."
            )
        if meta is None or len(meta) != self.batch_size:
            raise ValueError(
                "Metadata must be provided to unwrap tensors from multiple volumes."
            )

        tensors = []
        batch_size = data.batch_size // self.num_volumes
        coord_groups = None
        if data.coord_cols is not None:
            coord_groups = np.asarray(data.coord_cols).reshape(-1, 3)
        for batch_id in range(batch_size):
            tensor_list = []
            for volume_id in range(self.num_volumes):
                idx = batch_id * self.num_volumes + volume_id
                tensor = data[idx]
                if volume_id > 0 and coord_groups is not None:
                    for cols in coord_groups:
                        tensor[:, cols] = self.geo.translate(
                            tensor[:, cols], 0, volume_id, 1.0 / meta[batch_id].size
                        )
                if self.remove_batch_col and data.has_batch_col:
                    tensor = tensor[:, BATCH_COL + 1 :]

                tensor_list.append(tensor)

            tensors.append(np.concatenate(tensor_list))

        return tensors

    def _unwrap_index(
        self, data: Union[IndexBatch, EdgeIndexBatch]
    ) -> Union[List[np.ndarray], List[ObjectList]]:
        """Unwrap an index list into its constituents."""
        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            indexes = data.split()
        else:
            batch_size = data.batch_size // self.num_volumes
            indexes = []
            for batch_id in range(batch_size):
                index_list = []
                for volume_id in range(self.num_volumes):
                    idx = batch_id * self.num_volumes + volume_id
                    offset = (
                        data.offsets[idx] - data.offsets[batch_id * self.num_volumes]
                    )
                    index = data[idx]
                    if isinstance(data, IndexBatch) and data.is_list:
                        index_list.extend(offset + element for element in index)
                    else:
                        index_list.append(offset + index)

                if isinstance(data, IndexBatch) and data.is_list:
                    indexes.append(index_list)
                else:
                    indexes.append(np.concatenate(index_list))

        if isinstance(data, IndexBatch) and data.is_list:
            shape = (0, data.shape[1]) if len(data.shape) == 2 else 0
            default = np.empty(shape, dtype=np.int64)
            indexes_obl = []
            for index in indexes:
                indexes_obl.append(ObjectList(index, default=default))

            return indexes_obl

        return indexes

    def _unwrap_index_spans(self, data: Union[IndexBatch, EdgeIndexBatch]) -> List[int]:
        """Unwrap per-entry parent spans alongside index-like batches."""
        spans = data.spans
        if not isinstance(spans, np.ndarray):
            spans = spans.detach().cpu().numpy()

        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            return [int(span) for span in spans]

        batch_size = data.batch_size // self.num_volumes
        unwrapped_spans = []
        for batch_id in range(batch_size):
            lower = batch_id * self.num_volumes
            upper = (batch_id + 1) * self.num_volumes
            unwrapped_spans.append(int(np.sum(spans[lower:upper], dtype=np.int64)))

        return unwrapped_spans
