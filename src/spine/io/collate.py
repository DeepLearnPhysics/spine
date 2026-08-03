"""Contains implementations of data collation classes.

Collate classes are a middleware between parsers and datasets. They are given
to :class:`torch.utils.data.DataLoader` as the `collate_fn` argumement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    ObjectListBatch,
    ObjectListData,
    TensorBatch,
    TensorData,
)
from spine.geo import GeoManager

from .overlay import Overlayer

SampleDict = dict[str, Any]
BatchType = Sequence[SampleDict]

__all__ = ["CollateAll"]


class CollateAll:
    """General collate function for all data types coming from the parsers.

    Provide it with a list of dictionaries. Each value can be one of:

    - A `TensorData` with coordinates, features and metadata, merged into
      rows of the form `[batch_id, *coords, *features]`
    - A feature-only `TensorData`, merged into `[batch_id, *features]`
    - A `IndexData`, `IndexListData` or `EdgeIndexData`, merged into
      an offset-adjusted index batch
    - Scalar values, lists and objects, gathered into a list
    """

    name = "all"

    def __init__(
        self,
        data_keys: Sequence[str] | Mapping[str, Any] | None = None,
        split: bool = False,
        target_id: int = 0,
        source: Mapping[str, str] | None = None,
        overlay: Mapping[str, Any] | None = None,
        overlay_methods: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the collation parameters.

        Parameters
        ----------
        data_keys : sequence or mapping
            Names of products returned by the dataset. Mapping values are
            ignored and accepted only as a transition from the old API.
        split : bool, default False
            Whether to split the input by module ID (each module gets its
            own batch ID, multiplies the number of batches by `num_modules`)
        target_id : int, default 0
            If split is `True`, specifies where to relocate the points
        source : mapping, optional
            Mapping which maps keys to their corresponding sources. This can
            be used to split tensors without having to check the geometry
        overlay : mapping, optional
            Image overlay configuration
        overlay_methods : mapping
            Mapping of overlay methods
        """
        if data_keys is None:
            raise ValueError("Must provide the dataset `data_keys`.")
        self.data_keys = tuple(data_keys)

        # Initialize the geometry, if required
        self.split = split
        self.source = None
        if split:
            self.target_id = target_id
            self.geo = GeoManager.get_instance()
            self.num_modules = self.geo.tpc.num_modules
            self.source = source

        # Initialize the overlayer, if required
        self.overlayer = None
        if overlay is not None:
            if overlay_methods is None:
                raise ValueError(
                    "`overlay_methods` must be provided if `overlay` is not None."
                )
            self.overlayer = Overlayer(
                **overlay, data_keys=self.data_keys, methods=overlay_methods
            )

    def __call__(self, batch: BatchType) -> dict[str, Any]:
        """Takes a list of parsed information, one per event in a batch, and
        collates them into a single object per entry in the batch.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.

        Returns
        -------
        Dict
            Dictionary that matches one data key to one batch-worth of data
        """
        # Overlay data (modify batch), if needed
        if self.overlayer is not None:
            batch = self.overlayer(batch)

        # Loop over the data keys, merge all events in a batch
        data = {}
        for key in self.data_keys:
            # Dispatch directly on the self-describing event product
            ref_data = batch[0][key]
            if isinstance(ref_data, ClusterLabelData):
                data[key] = self.stack_cluster_labels(batch, key)

            elif isinstance(ref_data, TensorData):
                if ref_data.coordinate_data is not None and not ref_data.feats_only:
                    data[key] = self.stack_coord_tensors(batch, key)
                else:
                    data[key] = self.stack_feat_tensors(batch, key)

            elif isinstance(ref_data, (IndexData, IndexListData, EdgeIndexData)):
                data[key] = self.stack_index_tensors(batch, key)

            elif isinstance(ref_data, ObjectListData):
                data[key] = ObjectListBatch(sample[key] for sample in batch)

            else:
                # Scalars and self-defining physics objects are gathered
                data[key] = [sample[key] for sample in batch]

        return data

    def stack_cluster_labels(self, batch: BatchType, key: str) -> ClusterLabelBatch:
        """Collate event-level compact cluster labels.

        Parameters
        ----------
        batch : sequence of dict
            Parsed input events.
        key : str
            Cluster-label product key.

        Returns
        -------
        ClusterLabelBatch
            Batched voxel associations and optional particle table.
        """
        entries = [sample[key] for sample in batch]
        if not all(isinstance(entry, ClusterLabelData) for entry in entries):
            raise TypeError("Cluster-label batches require ClusterLabelData entries.")

        has_particles = entries[0].particles is not None
        if any((entry.particles is not None) != has_particles for entry in entries):
            raise ValueError("Particle information must be consistent across a batch.")

        # Build the compact batched voxel table.
        counts = np.asarray([len(entry.coords) for entry in entries], dtype=np.int64)
        coords = np.concatenate([entry.coords for entry in entries], axis=0)
        features = np.concatenate([entry.features for entry in entries], axis=0)
        batch_ids = np.repeat(np.arange(len(entries), dtype=coords.dtype), counts)
        data = np.concatenate((batch_ids[:, None], coords, features), axis=1)
        tensor = TensorBatch(
            data,
            counts,
            has_batch_col=True,
            coord_cols=np.arange(1, 4, dtype=np.int64),
            schema=ClusterLabelData.tensor_schema(has_particles),
            meta=[entry.meta for entry in entries],
        )

        # Stack every named particle field with shared event counts.
        particles = None
        if has_particles:
            reference_fields = tuple(entries[0].particles)
            if any(tuple(entry.particles) != reference_fields for entry in entries):
                raise ValueError(
                    "Particle-table fields must be consistent across a batch."
                )
            particle_counts = np.asarray(
                [len(next(iter(entry.particles.values()), ())) for entry in entries],
                dtype=np.int64,
            )
            particles = {}
            for name in reference_fields:
                values = np.concatenate(
                    [entry.particles[name] for entry in entries], axis=0
                )
                particles[name] = TensorBatch(values, particle_counts)

        return ClusterLabelBatch(
            tensor,
            particles,
            meta=[entry.meta for entry in entries],
        )

    def stack_coord_tensors(self, batch: BatchType, key: str) -> TensorBatch:
        """Stack coordinate tensors together across an overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Data product key

        Returns
        -------
        TensorBatch
            Batched coordinate tensor
        """
        # Enforce a common schema before combining physical coordinate columns
        reference = batch[0][key]
        if any(sample[key].schema != reference.schema for sample in batch[1:]):
            raise ValueError(f"Tensor schemas do not match for product `{key}`.")
        batch_size = len(batch)
        if not self.split:
            # Preserve one batch ID per input event in the unsplit case
            coords = np.vstack([sample[key].coordinate_data for sample in batch])
            features = np.vstack([sample[key].features for sample in batch])
            counts = [len(sample[key].coordinate_data) for sample in batch]
            batch_ids = np.repeat(np.arange(batch_size, dtype=coords.dtype), counts)

        else:
            # Split coordinates and create one batch ID per event-volume pair
            coords_v, features_v, batch_ids_v = [], [], []
            counts = np.empty(batch_size * self.num_modules, dtype=np.int64)
            for s, sample in enumerate(batch):
                # Relocate all coordinate groups into the target module frame
                coords = sample[key].coordinate_data
                features = sample[key].features
                meta = sample[key].meta
                coords_wrapped, module_indexes = self.geo.split(
                    coords.reshape(-1, 3), self.target_id, meta=meta
                )
                coords = coords_wrapped.reshape(-1, coords.shape[1])

                # Assign multi-point rows once when their points span volumes
                if coords.shape[1] > 3:
                    num_points = coords.shape[1] // 3
                    free = np.ones(len(coords), dtype=bool)
                    for m, module_index in enumerate(module_indexes):
                        mask = np.zeros(len(coords_wrapped), dtype=bool)
                        mask[module_index] = True
                        mask = mask.reshape(-1, num_points).any(axis=1)
                        module_indexes[m] = np.where(free & mask)[0]
                        free[module_indexes[m]] = False

                # Collect each volume under its flattened event-volume ID
                for m, module_index in enumerate(module_indexes):
                    coords_v.append(coords[module_index])
                    features_v.append(features[module_index])
                    idx = self.num_modules * s + m
                    batch_ids_v.append(
                        np.full(len(module_index), idx, dtype=coords.dtype)
                    )
                    counts[idx] = len(module_index)

            coords = np.vstack(coords_v)
            features = np.vstack(features_v)
            batch_ids = np.concatenate(batch_ids_v)

        # Assemble the self-describing batched tensor with shifted coordinate columns
        tensor = np.hstack([batch_ids[:, None], coords, features])
        coord_cols = np.arange(1, 1 + coords.shape[1])

        return TensorBatch(
            tensor.astype(features.dtype),
            counts,
            has_batch_col=True,
            coord_cols=coord_cols,
            schema=reference.schema,
            meta=(
                [sample[key].meta for sample in batch]
                if not self.split
                else [
                    sample[key].meta
                    for sample in batch
                    for _ in range(self.num_modules)
                ]
            ),
        )

    def stack_index_tensors(
        self, batch: BatchType, key: str
    ) -> IndexBatch | EdgeIndexBatch:
        """Stack index tensors together across an overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Data product key

        Returns
        -------
        Union[IndexBatch, EdgeIndexBatch]
            Batched index tensor
        """
        # Compute event offsets in the common node-index space
        total_counts = [sample[key].span for sample in batch]
        spans = np.asarray(total_counts, dtype=np.int64)
        offsets = np.zeros(len(total_counts), dtype=int)
        offsets[1:] = np.cumsum(total_counts)[:-1]

        if isinstance(batch[0][key], IndexListData):
            # Preserve each jagged member while shifting its node references
            index_list = []
            counts = []
            single_counts = []
            for i, sample in enumerate(batch):
                sample_index_list = [
                    np.asarray(index, dtype=np.int64) + offsets[i]
                    for index in sample[key].features
                ]
                index_list.extend(sample_index_list)
                counts.append(len(sample_index_list))
                if sample[key].single_counts is not None:
                    single_counts.extend(sample[key].single_counts.tolist())
                else:
                    single_counts.extend(len(index) for index in sample_index_list)

            return IndexBatch(index_list, spans, counts, single_counts)

        # Concatenate flat indexes or edge matrices without a batch column
        index_list = []
        for i, sample in enumerate(batch):
            index_list.append(sample[key].features + offsets[i])
        axis = 0 if index_list[0].ndim == 1 else 1
        index = np.concatenate(index_list, axis=axis)
        counts = [sample[key].features.shape[-1] for sample in batch]

        # Preserve the event-level subtype in the corresponding batch product
        if len(index.shape) == 1:
            return IndexBatch(index, spans, counts)
        else:
            return EdgeIndexBatch(index, counts, spans, directed=True)

    def stack_feat_tensors(self, batch: BatchType, key: str) -> TensorBatch:
        """Stack feature tensors together across an overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Data product key

        Returns
        -------
        TensorBatch
            Batched feature tensor
        """
        # Resolve optional module assignments for feature-only products
        sources = None
        if self.split and self.source is not None and key in self.source:
            source_key = self.source[key]
            sources = [batch[i][source_key].features for i in range(len(batch))]

        # Stack directly unless module splitting has a usable source product
        if not self.split or sources is None:
            tensor = np.concatenate([sample[key].features for sample in batch])
            counts = [len(sample[key].features) for sample in batch]

        else:
            # Partition features under the same flattened event-volume convention
            batch_size = len(batch)
            features_v = []
            counts = np.empty(batch_size * self.num_modules, dtype=np.int64)
            for s, sample in enumerate(batch):
                features = sample[key].features
                for m in range(self.num_modules):
                    module_index = np.where(sources[s][:, 0] == m)[0]
                    features_v.append(features[module_index])
                    idx = self.num_modules * s + m
                    counts[idx] = len(module_index)

            tensor = np.vstack(features_v)

        # Carry the common event schema into the resulting batch product
        reference = batch[0][key]
        if any(sample[key].schema != reference.schema for sample in batch[1:]):
            raise ValueError(f"Tensor schemas do not match for product `{key}`.")
        return TensorBatch(
            tensor,
            counts,
            schema=reference.schema,
            meta=[sample[key].meta for sample in batch],
        )
