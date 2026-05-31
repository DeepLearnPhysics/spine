"""Utilities for converting batched data structures into per-entry objects."""

from typing import Any

import numpy as np

from spine.constants import BATCH_COL
from spine.data import EdgeIndexBatch, IndexBatch, Meta, ObjectList, TensorBatch
from spine.geo import GeoManager

__all__ = ["Unwrapper"]


class Unwrapper:
    """Convert batched data structures into per-event entries.

    The `Unwrapper` is responsible for converting model input/output dictionaries
    containing batched tensors, indices, and metadata into a human-readable,
    per-event format. This is essential for post-processing, visualization, and
    evaluation, as model operations typically concatenate or stack data for
    efficient computation. The unwrapper restores the original event-wise
    structure, handling both single- and multi-volume (e.g., multi-TPC) data,
    and exporting auxiliary metadata such as per-entry spans.

    The main supported structures are `TensorBatch`, `IndexBatch`, and
    `EdgeIndexBatch`. Index-like batches also export per-entry span metadata
    under an additional `<name>_spans` key when it is not already present.
    """

    def __init__(self, meta_key: str = "meta", remove_batch_col: bool = False):
        """Initialize the unwrapper.

        Parameters
        ----------
        meta_key : str, optional
            Key in the input dictionary containing per-event metadata. This is
            used by multi-volume tensor unwrapping to translate coordinates.
        remove_batch_col : bool, optional
            If `True`, remove the batch index column from unwrapped tensors
            when it is present.
        """
        self.geo = GeoManager.get_instance_if_initialized()

        self.num_volumes = self.geo.tpc.num_modules if self.geo else 1
        self.meta_key = meta_key
        self.remove_batch_col = remove_batch_col
        self.batch_size = None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Unwrap a batched input/output dictionary into per-event entries.

        Parameters
        ----------
        data : dict
            Dictionary containing batched model inputs or outputs. Keys may
            include tensors, index batches, edge index batches, and optional
            metadata.

        Returns
        -------
        dict
            Dictionary with the same keys as input, but with each value unwrapped
            into per-event entries. For index-like batches, an additional
            `<name>_spans` key is added when absent.
        """
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

    def _unwrap(self, key: str, data: Any, meta: list[Meta] | None = None) -> Any:
        """Route one value to the appropriate unwrapping scheme.

        Parameters
        ----------
        key : str
            Name of the data field, used for error messages.
        data : Any
            Batched value to unwrap.
        meta : list[Meta], optional
            Per-event metadata, required for multi-volume unwrapping.

        Returns
        -------
        Any
            Unwrapped value, typically a list or ObjectList of per-event entries.

        Raises
        ------
        ValueError
            If the input is empty, the batch size is unset, or the type is
            unsupported.
        """
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
        self, data: TensorBatch, meta: list[Meta] | None = None
    ) -> list[Any]:
        """Unwrap a tensor batch into per-event tensors.

        Handles both single-volume and multi-volume data. For multi-volume
        tensors, coordinates may be translated to a common volume using the
        initialized geometry and per-entry metadata.

        Parameters
        ----------
        data : TensorBatch
            Batched tensor object to unwrap.
        meta : list[Meta], optional
            Per-event metadata, required for multi-volume unwrapping.

        Returns
        -------
        list
            Per-event tensors. Entries are usually `np.ndarray` objects, but
            they may follow the backend used by the input `TensorBatch` in the
            simple single-volume path.

        Raises
        ------
        ValueError
            If geometry or metadata is missing for multi-volume unwrapping.
        TypeError
            If multi-volume coordinate translation is requested on a
            non-numpy-backed tensor batch.
        """
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
                if not isinstance(tensor, np.ndarray):
                    raise TypeError(
                        "Multi-volume tensor unwrapping with geometry translation "
                        "requires a numpy-backed TensorBatch."
                    )

                if volume_id > 0 and coord_groups is not None:
                    for cols in coord_groups:
                        coord_cols = np.asarray(cols, dtype=np.int64)
                        translated_coords = self.geo.translate(
                            tensor[:, coord_cols],
                            0,
                            volume_id,
                            1.0 / meta[batch_id].size,
                        )
                        tensor[:, coord_cols] = translated_coords
                if self.remove_batch_col and data.has_batch_col:
                    tensor = tensor[:, BATCH_COL + 1 :]

                tensor_list.append(tensor)

            tensors.append(np.concatenate(tensor_list))

        return tensors

    def _unwrap_index(
        self, data: IndexBatch | EdgeIndexBatch
    ) -> list[np.ndarray] | list[ObjectList]:
        """Unwrap an index-like batch into per-event indexes.

        For multi-volume data, offsets are adjusted to produce event-local
        indexes. For `IndexBatch` objects with list structure, the result is
        wrapped in `ObjectList` containers.

        Parameters
        ----------
        data : IndexBatch or EdgeIndexBatch
            Batched index or edge index object to unwrap.

        Returns
        -------
        list[np.ndarray] or list[ObjectList]
            Per-event index arrays or object lists, matching the input
            structure.
        """
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
                object_list: list[object] = [element for element in index]
                indexes_obl.append(ObjectList(object_list, default=default))

            return indexes_obl

        return indexes

    def _unwrap_index_spans(self, data: IndexBatch | EdgeIndexBatch) -> list[int]:
        """Unwrap and export per-entry parent spans for index-like batches.

        For multi-volume data, sums the spans across all volumes for each event.
        For single-volume, returns the stored spans directly.

        Parameters
        ----------
        data : IndexBatch or EdgeIndexBatch
            Batched index or edge index object with span metadata.

        Returns
        -------
        list[int]
            List of per-event parent spans, one per event in the batch.
        """
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
