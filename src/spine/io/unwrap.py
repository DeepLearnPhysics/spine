"""Utilities for converting batched data structures into per-entry objects."""

from typing import Any

import numpy as np

from spine.data import (
    ClusterLabelBatch,
    ClusterLabelData,
    EdgeIndexBatch,
    EdgeIndexData,
    IndexBatch,
    IndexData,
    IndexListData,
    Meta,
    TensorBatch,
    TensorBatchConvertible,
    TensorData,
)
from spine.geo import GeoManager

__all__ = ["Unwrapper"]


class Unwrapper:
    """Convert batched data structures into per-event entries.

    The `Unwrapper` is responsible for converting model input/output dictionaries
    containing batched tensors, indices, and metadata into a human-readable,
    per-event format. This is essential for post-processing, visualization, and
    evaluation, as model operations typically concatenate or stack data for
    efficient computation. The unwrapper restores the original event-wise
    structure, handling both single- and multi-volume (e.g., multi-TPC) data.

    The main supported structures are `TensorBatch`, `IndexBatch`, and
    `EdgeIndexBatch`. Their output products are `TensorData`, `IndexData`,
    `IndexListData`, and `EdgeIndexData`; index spans therefore remain attached
    to the event product instead of appearing under parallel sidecar keys.
    """

    def __init__(self, meta_key: str = "meta"):
        """Initialize the unwrapper.

        Parameters
        ----------
        meta_key : str, optional
            Key in the input dictionary containing per-event metadata. This is
            used by multi-volume tensor unwrapping to translate coordinates.
        """
        # Capture optional geometry once, then derive the expected volume layout
        self.geo = GeoManager.get_instance_if_initialized()

        self.num_volumes = self.geo.tpc.num_modules if self.geo else 1
        self.meta_key = meta_key
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
            into self-describing per-event entries.
        """
        # Multi-volume coordinate translation requires event-aligned metadata
        meta = None
        if self.num_volumes > 1 and self.meta_key in data:
            meta = data[self.meta_key]

        # Establish the logical event axis before dispatching individual products
        data_unwrapped = {}
        self.batch_size = len(data["index"])
        for key, value in data.items():
            data_unwrapped[key] = self._unwrap(key, value, meta)

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
            Unwrapped value, typically a list of per-event data products.

        Raises
        ------
        ValueError
            If the input is empty, the batch size is unset, or the type is
            unsupported.
        """
        # Reject invalid containers before adapting supported product types.
        if isinstance(data, (list, tuple)) and len(data) == 0:
            raise ValueError(f"Batched data for {key} is an empty list, cannot unwrap.")
        if self.batch_size is None:
            raise ValueError("Batch size should be set before unwrapping.")

        # Structured labels share ordinary tensor geometry handling but retain
        # their compact particle tables on the final event products.
        if isinstance(data, ClusterLabelBatch):
            return self._unwrap_cluster_labels(data, meta)

        if isinstance(data, TensorBatchConvertible):
            data = data.to_tensor_batch()
        elif (
            isinstance(data, list)
            and len(data)
            and isinstance(data[0], TensorBatchConvertible)
        ):
            data = [value.to_tensor_batch() for value in data]

        # Scalars and ordinary event lists already use their final representation
        dim = len(getattr(data, "shape", (0,)))
        if (
            np.isscalar(data)
            or dim == 0
            or (isinstance(data, list) and not isinstance(data[0], TensorBatch))
        ):
            return data

        if isinstance(data, TensorBatch):
            return self._unwrap_tensor(data, meta)

        # Parallel tensor outputs become one list of tensors per logical event
        if isinstance(data, list) and isinstance(data[0], TensorBatch):
            data_split = [self._unwrap_tensor(t, meta) for t in data]
            tensor_lists = []
            for batch_id in range(self.batch_size):
                tensor_lists.append([value[batch_id] for value in data_split])

            return tensor_lists

        if isinstance(data, (IndexBatch, EdgeIndexBatch)):
            return self._unwrap_index(data)

        raise ValueError(f"Type of {key} not unwrappable: {type(data)}")

    def _unwrap_cluster_labels(
        self,
        data: ClusterLabelBatch,
        meta: list[Meta] | None = None,
    ) -> list[ClusterLabelData]:
        """Unwrap compact cluster labels and their particle tables.

        Multi-volume collation duplicates each event particle table for every
        physical module while keeping voxel particle indexes event-local. The
        voxel rows follow the ordinary tensor geometry path; one copy of the
        identical particle table is then attached to each merged logical event.

        Parameters
        ----------
        data : ClusterLabelBatch
            Batched compact voxel labels and optional particle fields.
        meta : list[Meta], optional
            Metadata for each logical event.

        Returns
        -------
        list[ClusterLabelData]
            Structured labels on the logical event domain.
        """
        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            return [data[batch_id] for batch_id in range(data.batch_size)]

        tensors = self._unwrap_tensor(data.data, meta)
        labels = []
        for batch_id, tensor in enumerate(tensors):
            source_id = batch_id * self.num_volumes
            particles = None
            if data.particles is not None:
                particles = {
                    name: field[source_id] for name, field in data.particles.items()
                }
            event_meta = None if data.meta is None else data.meta[source_id]
            if meta is not None:
                event_meta = meta[batch_id]
            labels.append(
                ClusterLabelData(
                    coords=tensor.coords,
                    features=tensor.features,
                    particles=particles,
                    meta=event_meta,
                )
            )

        return labels

    def _unwrap_tensor(
        self, data: TensorBatch, meta: list[Meta] | None = None
    ) -> list[TensorData]:
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
        list[TensorData]
            Self-describing event tensors with batch columns removed.

        Raises
        ------
        ValueError
            If geometry or metadata is missing for multi-volume unwrapping.
        TypeError
            If multi-volume coordinate translation is requested on a
            non-numpy-backed tensor batch.
        """
        # Single-volume batches already expose the desired event partition
        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            return [data.event(batch_id) for batch_id in range(data.batch_size)]

        if self.geo is None:
            raise ValueError(
                "Geometry must be initialized to unwrap tensors from multiple volumes."
            )
        if meta is None or len(meta) != self.batch_size:
            raise ValueError(
                "Metadata must be provided to unwrap tensors from multiple volumes."
            )

        # Group physical volumes by logical event and identify coordinate triplets
        tensors = []
        batch_size = data.batch_size // self.num_volumes
        if data.data.ndim == 1 and data.coord_cols is not None:
            raise ValueError(
                "One-dimensional tensor products cannot carry coordinates."
            )
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

                # Translate secondary volumes into the reference module frame
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
                tensor_list.append(tensor)

            # Merge volumes and recover logical coordinate/feature matrices
            packed = np.concatenate(tensor_list)
            event_meta = None if data.meta is None else data.meta[batch_id]

            # Scalar feature batches remain one-dimensional after volumes are
            # merged. Preserve that event representation just as `event()`
            # does in the single-volume path.
            if packed.ndim == 1:
                tensors.append(TensorData(packed, meta=event_meta, schema=data.schema))
                continue

            coord_cols = np.asarray(
                () if data.coord_cols is None else data.coord_cols,
                dtype=np.int64,
            )
            excluded = set(coord_cols.tolist())
            if data.has_batch_col:
                excluded.add(data.batch_col)
            feature_cols = [
                column for column in range(packed.shape[1]) if column not in excluded
            ]
            coords = None if len(coord_cols) == 0 else packed[:, coord_cols]

            # Rebuild one self-describing tensor for the logical event
            tensors.append(
                TensorData(
                    packed[:, feature_cols],
                    coords,
                    meta=event_meta,
                    schema=data.schema,
                )
            )

        return tensors

    def _unwrap_index(
        self, data: IndexBatch | EdgeIndexBatch
    ) -> list[IndexData | IndexListData | EdgeIndexData]:
        """Unwrap an index-like batch into per-event indexes.

        For multi-volume data, offsets are adjusted to produce event-local
        indexes.

        Parameters
        ----------
        data : IndexBatch or EdgeIndexBatch
            Batched index or edge index object to unwrap.

        Returns
        -------
        list[IndexData or IndexListData or EdgeIndexData]
            Self-describing event index products matching the input structure.
        """
        # Single-volume index products already carry event-local namespaces
        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            return [data.event(batch_id) for batch_id in range(data.batch_size)]

        # Merge physical-volume namespaces into one namespace per logical event
        batch_size = data.batch_size // self.num_volumes
        indexes = []
        for batch_id in range(batch_size):
            index_list = []
            for volume_id in range(self.num_volumes):
                idx = batch_id * self.num_volumes + volume_id
                offset = data.offsets[idx] - data.offsets[batch_id * self.num_volumes]
                index = data[idx]
                if isinstance(data, IndexBatch) and data.is_list:
                    index_list.extend(offset + element for element in index)
                else:
                    index_list.append(offset + index)

            if isinstance(data, IndexBatch) and data.is_list:
                indexes.append(index_list)
            else:
                indexes.append(np.concatenate(index_list))

        # Restore the concrete event product and its combined parent span
        if isinstance(data, IndexBatch) and data.is_list:
            return [
                IndexListData(list(index), int(span))
                for index, span in zip(indexes, self._unwrap_index_spans(data))
            ]

        spans = self._unwrap_index_spans(data)
        if isinstance(data, EdgeIndexBatch):
            return [
                EdgeIndexData(index.T, int(span), data.directed)
                for index, span in zip(indexes, spans)
            ]
        return [IndexData(index, int(span)) for index, span in zip(indexes, spans)]

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
        # Normalize backend metadata before grouping physical-volume spans
        spans = data.spans
        if not isinstance(spans, np.ndarray):
            spans = spans.detach().cpu().numpy()

        if self.num_volumes == 1 or data.batch_size == self.batch_size:
            return [int(span) for span in spans]

        # Each logical span is the sum of its constituent volume namespaces
        batch_size = data.batch_size // self.num_volumes
        unwrapped_spans = []
        for batch_id in range(batch_size):
            lower = batch_id * self.num_volumes
            upper = (batch_id + 1) * self.num_volumes
            unwrapped_spans.append(int(np.sum(spans[lower:upper], dtype=np.int64)))

        return unwrapped_spans
