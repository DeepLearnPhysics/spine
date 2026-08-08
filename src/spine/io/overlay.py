"""Module with methods to overlay multiple events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from warnings import warn

import numpy as np

from spine.data import (
    ClusterLabelData,
    DataBase,
    EdgeIndexData,
    IndexData,
    IndexListData,
    ObjectListData,
    TensorData,
)

from .parse.clean_data import clean_sparse_data

SampleDict = dict[str, Any]
BatchType = Sequence[SampleDict]

__all__ = ["Overlayer"]


class Overlayer:
    """Generic class to produce data overlays.

    This class supports three overlay modes:

    - `constant` uses a fixed multiplicity.
    - `uniform` samples multiplicities `M_i` from a uniform distribution and
      adjusts them so that, for a batch size `B`, `sum_i M_i = B`.
    - `poisson` samples multiplicities from a Poisson distribution with mean
      set by `multiplicity` and adjusts them the same way.
    """

    # List of recognized overlay modes
    _modes = ("constant", "uniform", "poisson")

    def __init__(
        self,
        data_keys: Sequence[str] | Mapping[str, Any] | None,
        methods: Mapping[str, str | None],
        multiplicity: int,
        mode: str = "constant",
    ) -> None:
        """Store the overlay parameters.

        Parameters
        ----------
        data_keys : sequence or mapping
            Names of products returned by the upstream dataset.
        methods : mapping
            Maps data products onto overlay methods
        multiplicity : int
            Number of images to stack in the overlay
        mode : str, default 'constant'
            Overlay mode (one of 'constant', 'uniform' or 'poisson')
        """
        # Check that the overlay mode is recognized
        if mode not in self._modes:
            raise ValueError(
                f"Overlay mode not recognized: {mode}. Must be one of {self._modes}."
            )
        self.mode = mode

        # Check that multiplicity is sensible
        if multiplicity <= 0:
            raise ValueError(
                "Overlay multiplicity should be a non-zero positive integer."
            )
        self.multiplicity = multiplicity

        if data_keys is None:
            raise ValueError("Must provide the dataset `data_keys`.")
        self.data_keys = tuple(data_keys)
        self.methods = methods

        # Initialize row selection references for feature-only tensors
        self._row_selections = {}

    def __call__(self, batch: BatchType) -> list[SampleDict]:
        """Given a batch of data, provides an overlay batching and modifies
        the data in place to avoid indexing conflicts.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.

        Returns
        -------
        List[Dict]
            Overlayed list of dictionaries of parsed information, one per overlay.
        """
        # Fetch the batch size, build an overlap map
        batch_size = len(batch)
        overlay_ids = self.get_assignments(batch_size)

        # Loop over the unique overlay indexes
        overlay_batch = []
        _, splits = np.unique(overlay_ids, return_index=True)
        indexes = np.split(np.arange(batch_size), splits[1:])
        for index in indexes:
            # Initialize row selection references for feature-only tensors
            self._row_selections = {}

            # If there is only a single index in the overlay, nothing to do
            if len(index) < 2:
                overlay_batch.append(batch[index[0]])
                continue

            # Loop over the keys to overlay
            overlay = {}
            for key in self.get_overlay_order(batch, index):
                # Dispatch directly on the event product
                reference = batch[index[0]][key]
                if np.isscalar(reference) or isinstance(reference, str):
                    # Check whether scalars can be harmonized
                    overlay[key] = self.merge_scalars(batch, key, index)

                elif isinstance(reference, ObjectListData):
                    # Offset object list index attributes if needed
                    overlay[key] = self.cat_objects(batch, key, index)

                elif isinstance(reference, DataBase):
                    # Check that objects are compatible when overlaying
                    overlay[key] = self.merge_objects(batch, key, index)

                elif isinstance(
                    reference, (TensorData, IndexData, IndexListData, EdgeIndexData)
                ):
                    # Stack tensors, offset index columns if needed
                    overlay[key] = self.stack_tensors(batch, key, index)

                elif isinstance(reference, ClusterLabelData):
                    overlay[key] = self.stack_cluster_labels(batch, key, index)

                else:
                    # Scalar-like Python and user data objects need no parser
                    # taxonomy: their configured merge policy is sufficient.
                    overlay[key] = self.merge_objects(batch, key, index)

            # Add overlay to the batch
            overlay_batch.append(overlay)

        return overlay_batch

    def stack_cluster_labels(
        self, batch: BatchType, key: str, index: np.ndarray | Sequence[int]
    ) -> ClusterLabelData:
        """Overlay compact cluster labels and their particle tables.

        Parameters
        ----------
        batch : sequence of dict
            Event dictionaries containing the cluster-label product.
        key : str
            Cluster-label product key to overlay.
        index : np.ndarray or sequence[int]
            Positions of the source events included in this overlay.

        Returns
        -------
        ClusterLabelData
            Merged voxel labels with disjoint cluster and particle namespaces.

        Raises
        ------
        TypeError
            If any selected entry is not a :class:`ClusterLabelData` product.
        ValueError
            If spatial metadata or particle-table availability differs between
            the selected entries.
        """
        entries = [batch[idx][key] for idx in index]
        reference = entries[0]
        if not all(isinstance(entry, ClusterLabelData) for entry in entries):
            raise TypeError("Cluster-label overlay requires matching parser products.")
        if not all(entry.meta == reference.meta for entry in entries):
            raise ValueError("Cluster-label metadata must match across an overlay.")
        has_particles = reference.particles is not None
        if any((entry.particles is not None) != has_particles for entry in entries):
            raise ValueError(
                "Particle information must be consistent across an overlay."
            )
        precedence = reference.precedence
        if any(
            not np.array_equal(entry.precedence, precedence) for entry in entries[1:]
        ):
            raise ValueError("Shape precedence must be consistent across an overlay.")

        # Track independent namespaces while concatenating source events
        coords_list = []
        feature_list = []
        shape_list = []
        particle_tables = []
        cluster_offset = 0
        particle_table_offset = 0
        particle_id_offset = 0
        group_offset = 0
        interaction_offset = 0
        neutrino_offset = 0
        has_precedence = precedence is not None
        for entry in entries:
            # Cluster IDs index voxel-level instances
            features = entry.features.copy()
            valid_cluster = features[:, 1] >= 0
            features[valid_cluster, 1] += cluster_offset
            cluster_offset += int(np.max(entry.features[:, 1], initial=-1)) + 1

            if has_particles:
                if has_precedence:
                    shape_list.append(entry.shapes)

                # Associations and ancestors index rows in the particle table
                valid_particle = features[:, 2] >= 0
                features[valid_particle, 2] += particle_table_offset
                table = {name: value.copy() for name, value in entry.particles.items()}

                # Physical particle/group/interaction IDs have distinct namespaces
                particle_id_count = int(np.max(table["particle"], initial=-1)) + 1
                valid = table["particle"] >= 0
                table["particle"][valid] += particle_id_offset
                group_count = int(np.max(table["group"], initial=-1)) + 1
                valid = table["group"] >= 0
                table["group"][valid] += group_offset
                valid = table["ancestor"] >= 0
                table["ancestor"][valid] += particle_table_offset
                interaction_count = int(np.max(table["interaction"], initial=-1)) + 1
                valid = table["interaction"] >= 0
                table["interaction"][valid] += interaction_offset
                neutrino_count = int(np.max(table["nu"], initial=-1)) + 1
                valid = table["nu"] >= 0
                table["nu"][valid] += neutrino_offset
                particle_table_offset += len(table["particle"])
                particle_id_offset += particle_id_count
                group_offset += group_count
                interaction_offset += interaction_count
                neutrino_offset += neutrino_count
                particle_tables.append(table)

            coords_list.append(entry.coords)
            feature_list.append(features)

        # Merge overlapping voxels only after all index spaces are disjoint
        coords = np.concatenate(coords_list, axis=0)
        features = np.concatenate(feature_list, axis=0)
        prec_col = None
        if has_precedence:
            # Expand the carried precedence field only for duplicate selection.
            shapes = np.concatenate(shape_list, axis=0).reshape(-1, 1)
            shapes = shapes.astype(features.dtype, copy=False)
            features = np.concatenate((features, shapes), axis=1)
            prec_col = features.shape[1] - 1
        coords, features, selection = clean_sparse_data(
            coords,
            features,
            sum_cols=np.asarray([0], dtype=np.int64),
            prec_col=prec_col,
            precedence=precedence,
            return_index=True,
        )
        if has_precedence:
            features = features[:, :-1]
        self._row_selections[key] = (selection, sum(len(x) for x in feature_list))

        # Reassemble the particle side of the structured product
        particles = None
        if has_particles:
            particles = {
                name: np.concatenate([table[name] for table in particle_tables], axis=0)
                for name in particle_tables[0]
            }
        return ClusterLabelData(
            coords=coords,
            features=features,
            particles=particles,
            meta=reference.meta,
            sum_cols=reference.sum_cols,
            precedence=precedence,
        )

    def get_overlay_order(
        self, batch: BatchType, index: np.ndarray | Sequence[int]
    ) -> list[str]:
        """Order reference tensors before tensors that depend on them.

        Feature-only tensors such as source IDs may be row-aligned to another
        tensor that drops duplicate coordinates during overlay. Processing
        references first lets them define the row selection reused by aligned
        tensors.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        index : np.ndarray or Sequence[int]
            List of indexes to merge into an overlay

        Returns
        -------
        List[str]
            List of keys in the order they should be processed for overlay.
        """
        ordered = []
        visited = set()
        visiting = set()

        def visit(key: str) -> None:
            if key in visited:
                return
            if key in visiting:
                raise ValueError(f"Cyclic overlay reference involving `{key}`.")

            ref_data = batch[index[0]][key]
            if isinstance(ref_data, TensorData) and ref_data.overlay_reference:
                reference = ref_data.overlay_reference
                if reference not in self.data_keys:
                    raise ValueError(
                        f"Overlay reference `{reference}` for `{key}` is not "
                        "available in the overlaid products."
                    )
                visiting.add(key)
                visit(reference)
                visiting.remove(key)

            visited.add(key)
            ordered.append(key)

        for key in self.data_keys:
            visit(key)

        return ordered

    def get_assignments(self, batch_size: int) -> np.ndarray:
        """Given a data product count, produce batch assignments.

        Parameters
        ----------
        batch_size : int
            Number of entries in the batch

        Returns
        -------
        np.ndarray
            Overlay ID assignments
        """
        # Dispatch
        if self.mode == "constant":
            # Uniform multiplicity of overlays
            if batch_size % self.multiplicity != 0:
                warn(
                    f"The overlay multiplicity ({self.multiplicity}) is not a "
                    f"divider of the batch size ({batch_size}). The overlay "
                    "multiplicity will not be uniform."
                )

            overlay_ids = np.arange(batch_size, dtype=int) // self.multiplicity

        elif self.mode in ("poisson", "uniform"):
            # Sample from a Poisson distribution until it adds up to the batch size
            overlay_ids = np.empty(batch_size, dtype=int)
            idx, total = 0, 0
            while total < batch_size:
                # Sample distribution
                if self.mode == "poisson":
                    sample = np.random.poisson(self.multiplicity)
                else:
                    sample = np.random.randint(1, self.multiplicity + 1)

                # Assign overlay indices
                if sample > 0:
                    overlay_ids[total : total + sample] = idx
                    idx += 1
                    total += sample

        else:
            raise ValueError(f"Overlay mode not recognized: {self.mode}.")

        # Return
        return overlay_ids

    def merge_scalars(
        self, batch: BatchType, key: str, index: np.ndarray | Sequence[int]
    ) -> Any:
        """Merge scalars into one per overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Scalar data product key
        index : np.ndarray
            List of indexes to merge into an overlay

        Returns
        -------
        object
            Single scalar for the batch
        """
        scalars = np.array([batch[idx][key] for idx in index])
        if self.methods[key] in ["first", "match"]:
            # Make sure that all scalars match within the overlay, if needed
            if self.methods[key] == "match":
                if not np.all(scalars[1:] == scalars[0]):
                    raise ValueError(
                        f"The scalar values to overlay do not match for {key}."
                    )

            return scalars[0]

        elif self.methods[key] == "sum":
            # Sum the values within each overlay
            return np.sum(scalars)

        elif self.methods[key] == "cat":
            # Concatenate the scalars in a single array (type change)
            return scalars

        else:
            if self.methods[key] is None:
                raise ValueError(f"Scalar overlay method not specified for {key}.")

            raise ValueError(
                f"Scalar overlay method not recognized: {self.methods[key]}. "
                "Must be one of 'first', 'match' or 'sum'."
            )

    def merge_objects(
        self, batch: BatchType, key: str, index: np.ndarray | Sequence[int]
    ) -> Any:
        """Merge objects into one per overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Object data product key
        index : np.ndarray
            List of indexes to merge into an overlay

        Returns
        -------
        object
            Single object for the batch
        """
        objects = [batch[idx][key] for idx in index]
        if self.methods[key] in ["first", "match"]:
            # Make sure that all objects match within the overlay, if needed
            if self.methods[key] == "match":
                if not np.all([obj == objects[0] for obj in objects]):
                    raise ValueError(f"The objects to overlay do not match for {key}.")

            return objects[0]

        elif self.methods[key] == "cat":
            # Concatenate the objects in a single list (type change)
            return ObjectListData(objects, default=objects[0])

        else:
            if self.methods[key] is None:
                raise ValueError(f"Object overlay method not specified for {key}.")

            raise ValueError(
                f"Object overlay method not recognized: {self.methods[key]}. "
                "Must be one of 'first' or 'match'."
            )

    def cat_objects(
        self, batch: BatchType, key: str, index: np.ndarray | Sequence[int]
    ) -> ObjectListData:
        """Concatenate object lists into one, offset index attributes if needed.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Object list data product key
        index : np.ndarray
            List of indexes to merge into an overlay

        Returns
        -------
        ObjList
            Concatenated obejct list
        """
        # If the objects in the lists contain indexes, must offset them
        ref_list = batch[index[0]][key]
        shifts = None
        if len(ref_list.default.index_attrs) > 0:
            shifts = ref_list.index_shifts
            if isinstance(shifts, dict):
                shifts = dict(shifts)
            for idx in index[1:]:
                # Shift indexes in the objects
                obj_list = batch[idx][key]
                for obj in obj_list:
                    obj.shift_indexes(shifts)

                # Increment shifts
                if not isinstance(shifts, dict):
                    shifts += obj_list.index_shifts
                else:
                    for attr in shifts:
                        shifts[attr] += obj_list.index_shifts[attr]

        # Concatenate and return
        obj_list = []
        for idx in index:
            obj_list.extend(batch[idx][key])

        return ObjectListData(obj_list, ref_list.default, shifts)

    def stack_tensors(
        self, batch: BatchType, key: str, index: np.ndarray | Sequence[int]
    ) -> TensorData | IndexData | IndexListData | EdgeIndexData:
        """Stack parser payloads together across an overlay.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Tensor data product key
        index : np.ndarray
            List of indexes to merge into an overlay

        Returns
        -------
        TensorData or IndexData or IndexListData or EdgeIndexData
            Overlayed parser payload of the same logical type as the input.
        """
        # Define a reference tensor
        ref_data = batch[index[0]][key]

        if isinstance(ref_data, TensorData):
            if ref_data.feats_only:
                return self.stack_feature_tensor_data(batch, key, index, ref_data)
            return self.stack_tensor_data(batch, key, index, ref_data)

        if isinstance(ref_data, IndexData):
            return self.stack_flat_index_data(batch, key, index, ref_data)

        if isinstance(ref_data, IndexListData):
            return self.stack_index_list_data(batch, key, index, ref_data)

        if isinstance(ref_data, EdgeIndexData):
            return self.stack_edge_index_data(batch, key, index, ref_data)

        raise TypeError(
            f"Unsupported parser payload type for `{key}`: {type(ref_data).__name__}."
        )

    def stack_tensor_data(
        self,
        batch: BatchType,
        key: str,
        index: np.ndarray | Sequence[int],
        ref_data: TensorData,
    ) -> TensorData:
        """Overlay one tensor-like parser payload.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Tensor data product key
        index : np.ndarray
            List of indexes to merge into an overlay
        ref_data : TensorData
            Reference tensor used to check metadata and index columns, and to
            preserve overlay metadata in the output.

        Returns
        -------
        TensorData
            Overlayed parser tensor
        """
        # Stack coordinates, if present
        coords = None
        if ref_data.coordinate_data is not None:
            # Check that the meta data matches between all images (it must)
            if not np.all([batch[idx][key].meta == ref_data.meta for idx in index]):
                raise ValueError("The metadata must match across all overlayed tensor.")
            coords = np.vstack([batch[idx][key].coordinate_data for idx in index])

        # If required, offset indexes in the feature tensor
        index_shifts = None
        if ref_data.index_cols is not None:
            # Apply offsets to the relevant columns only (mixed features)
            if ref_data.index_shifts is None:
                raise ValueError(
                    "Index shifts must be provided if index columns are present."
                )
            index_shifts = ref_data.index_shifts.copy()
            for idx in index[1:]:
                for i, col in enumerate(ref_data.index_cols):
                    mask = batch[idx][key].features[:, col] > -1
                    batch[idx][key].features[mask, col] += index_shifts[i]
                index_shifts += batch[idx][key].index_shifts

        # Stack features
        features = np.vstack([batch[idx][key].features for idx in index])

        # If requested, remove rows corresponding to duplicate coordinates
        if ref_data.remove_duplicates:
            # Check that we have coordinates to make the check
            if coords is None:
                raise ValueError("Must provide coordinates to filter duplicates.")

            # Filter out duplicates, aggregating features when requested.
            selection_size = len(features)
            coords, features, selection = clean_sparse_data(
                coords,
                features,
                sum_cols=ref_data.sum_cols,
                avg_cols=ref_data.avg_cols,
                prec_col=ref_data.prec_col,
                precedence=ref_data.precedence,
                return_index=True,
            )
            self._row_selections[key] = (selection, selection_size)

        return self.build_parser_tensor(ref_data, features, coords, index_shifts)

    def stack_feature_tensor_data(
        self,
        batch: BatchType,
        key: str,
        index: np.ndarray | Sequence[int],
        ref_data: TensorData,
    ) -> TensorData:
        """Overlay one feature-only parser payload.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Tensor data product key
        index : np.ndarray
            List of indexes to merge into an overlay
        ref_data : TensorData
            Reference tensor used to check metadata and index columns, and to
            preserve overlay metadata in the output.

        Returns
        -------
        TensorData
            Overlayed parser tensor with feature-only coordinates
        """
        # Stack the features
        features = np.vstack([batch[idx][key].features for idx in index])

        # Nothing to do if no duplicate removal is requested
        if not ref_data.remove_duplicates:
            return self.build_parser_tensor(ref_data, features, feats_only=True)

        # If it is requested, we need a reference tensor
        if not ref_data.overlay_reference:
            raise ValueError(
                f"Feature-only tensor `{key}` requires an `overlay_reference` "
                "to remove duplicates during overlay."
            )

        # Feature-only tensors reuse the duplicate policy of their reference.
        row_selection, row_selection_size = self._row_selections.get(
            ref_data.overlay_reference, (None, None)
        )
        if row_selection is None:
            # If the reference tensor has not been cleaned up, nothing to do
            # for the feature-only tensor either.
            return self.build_parser_tensor(ref_data, features, feats_only=True)

        if len(features) != row_selection_size:
            # If the sizes disagree, that is not allowed
            raise ValueError(
                f"Feature-only tensor `{key}` has {len(features)} rows before "
                f"overlay cleanup, but its reference `{ref_data.overlay_reference}` "
                f"has {row_selection_size} rows."
            )

        return self.build_parser_tensor(
            ref_data, features[row_selection], feats_only=True
        )

    @staticmethod
    def build_parser_tensor(
        ref_data: TensorData,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        index_shifts: np.ndarray | None = None,
        feats_only: bool | None = None,
    ) -> TensorData:
        """Build a parser tensor while preserving overlay metadata.

        Parameters
        ----------
        ref_data : TensorData
            Reference tensor used to check metadata and index columns, and to
            preserve overlay metadata in the output.
        features : np.ndarray
            Stacked features for the overlay
        coords : np.ndarray, optional
            Stacked coordinates for the overlay, if present in the reference tensor
        index_shifts : np.ndarray, optional
            Stacked index shifts for the overlay, if present in the reference tensor
        feats_only : bool, optional
            Whether the output tensor should be feature-only. If not provided, will
            be inferred from the reference tensor.

        Returns
        -------
        TensorData
            Overlay tensor carrying the reference product's schema and metadata.
        """
        return TensorData(
            coords=coords,
            features=features,
            meta=ref_data.meta,
            index_shifts=index_shifts,
            schema=ref_data.schema,
        )

    def stack_flat_index_data(
        self,
        batch: BatchType,
        key: str,
        index: np.ndarray | Sequence[int],
        ref_data: IndexData,
    ) -> IndexData:
        """Overlay one flat index payload.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Index data product key
        index : np.ndarray
            List of indexes to merge into an overlay
        ref_data : IndexData
            Reference index used to check metadata and preserve overlay metadata in
            the output.

        Returns
        -------
        IndexData
            Overlayed index data.
        """
        span = ref_data.span
        shifted_indexes = [batch[index[0]][key].features]
        for idx in index[1:]:
            shifted_index = batch[idx][key].features.copy()
            mask = shifted_index > -1
            shifted_index[mask] += span
            shifted_indexes.append(shifted_index)
            span += batch[idx][key].span

        features = np.concatenate(shifted_indexes, axis=-1)
        return IndexData(features=features, span=span)

    def stack_index_list_data(
        self,
        batch: BatchType,
        key: str,
        index: np.ndarray | Sequence[int],
        ref_data: IndexListData,
    ) -> IndexListData:
        """Overlay one jagged index-list payload.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Index list data product key
        index : np.ndarray
            List of indexes to merge into an overlay
        ref_data : IndexListData
            Reference index list used to check metadata and preserve overlay metadata
            in the output.

        Returns
        -------
        IndexListData
            Overlayed index list data.
        """
        span = ref_data.span
        features = [entry.copy() for entry in batch[index[0]][key].features]
        single_counts = []
        if ref_data.single_counts is not None:
            single_counts.extend(ref_data.single_counts.tolist())
        else:
            single_counts.extend(len(entry) for entry in features)

        for idx in index[1:]:
            shifted_entries = []
            for entry in batch[idx][key].features:
                shifted_entry = entry.copy()
                mask = shifted_entry > -1
                shifted_entry[mask] += span
                shifted_entries.append(shifted_entry)
            features.extend(shifted_entries)
            if batch[idx][key].single_counts is not None:
                single_counts.extend(batch[idx][key].single_counts.tolist())
            else:
                single_counts.extend(len(entry) for entry in shifted_entries)
            span += batch[idx][key].span

        return IndexListData(
            features=features,
            span=span,
            single_counts=np.asarray(single_counts, dtype=np.int64),
        )

    def stack_edge_index_data(
        self,
        batch: BatchType,
        key: str,
        index: np.ndarray | Sequence[int],
        ref_data: EdgeIndexData,
    ) -> EdgeIndexData:
        """Overlay one edge-index payload.

        Parameters
        ----------
        batch : List[Dict]
            List of dictionaries of parsed information, one per event. Each
            dictionary matches one data key to one event-worth of parsed data.
        key : str
            Edge index data product key
        index : np.ndarray
            List of indexes to merge into an overlay
        ref_data : EdgeIndexData
            Reference edge index used to check metadata and preserve overlay metadata
            in the output.

        Returns
        -------
        EdgeIndexData
            Overlayed edge index data.
        """
        span = ref_data.span
        shifted_indexes = [batch[index[0]][key].features]
        for idx in index[1:]:
            shifted_index = batch[idx][key].features.copy()
            mask = shifted_index > -1
            shifted_index[mask] += span
            shifted_indexes.append(shifted_index)
            span += batch[idx][key].span

        features = np.concatenate(shifted_indexes, axis=-1)
        return EdgeIndexData(features=features, span=span)
