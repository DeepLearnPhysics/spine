"""Module that contains all parsers related to LArCV cluster data.

Contains the following parsers:
- :class:`LArCVCluster2DParser`
- :class:`LArCVCluster3DParser`
- :class:`LArCVCluster3DAggregateParser`
- :class:`LArCVCluster3DChargeRescaledParser`
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from warnings import warn

import numpy as np

from spine.constants import LOWES_SHP, SHAPE_PREC
from spine.data import ClusterLabelData, Meta, TensorData
from spine.math.cluster import dbscan
from spine.math.distance import METRICS
from spine.utils.conditional import larcv
from spine.utils.particles import process_particle_event
from spine.utils.ppn import image_coordinates_batch

from ..base import ParserBase
from ..clean_data import clean_sparse_data
from .sparse import (
    LArCVSparse3DAggregateParser,
    LArCVSparse3DChargeRescaledParser,
    LArCVSparse3DParser,
)

__all__ = [
    "LArCVCluster2DParser",
    "LArCVCluster3DParser",
    "LArCVCluster3DAggregateParser",
    "LArCVCluster3DChargeRescaledParser",
]


class LArCVCluster2DParser(ParserBase):
    """Class that retrieves and parses a 2D cluster list.

    .. code-block. yaml

        schema:
          cluster_label:
            parser: cluster2d
            cluster_event: cluster2d_pcluster
    """

    # Name of the parser (as specified in the configuration)
    name = "cluster2d"

    # Type of object(s) returned by the parser

    def __init__(self, dtype: str, cluster_event: str, projection_id: int) -> None:
        """Initialize the parser.

        Parameters
        ----------
        cluster_event : larcv.EventClusterPixel2D
            Event which contains the 2D clusters
        projection_id : int
            Projection ID to get the 2D images from
        """
        # Initialize the parent class
        super().__init__(dtype, cluster_event=cluster_event)

        # Store the relevant attributes
        self.projection_id = projection_id

        # Define the overlay strategy parameters
        self.index_cols = np.array([1], dtype=np.int64)
        self.sum_cols = np.array([0], dtype=np.int64)

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object

        Returns
        -------
        TensorData
            Sparse tensor containing the parsed 2D cluster assignments.
        """
        return self.process(**self.get_input_data(trees))

    def process(self, cluster_event: Any) -> TensorData:
        """Converts a 2D clusters tensor into a single tensor.

        Parameters
        ----------
        cluster_event : larcv.EventClusterPixel2D
            Event which contains the 2D clusters

        Returns
        -------
        TensorData
            coords : np.ndarray
                (N, 2) array of [x, y] coordinates
            features : np.ndarray
                (N, 2) array of [pixel value, cluster ID]
            meta : Meta
                Metadata of the parsed image
        """
        # Get the cluster from the appropriate projection
        cluster_event_p = cluster_event.cluster_pixel_2d(self.projection_id)

        meta = cluster_event_p.meta()
        num_clusters = cluster_event_p.size()
        clusters = list(cluster_event_p.as_vector())
        cluster_sizes = np.fromiter(
            (cluster.as_vector().size() for cluster in clusters),
            dtype=np.int64,
            count=num_clusters,
        )
        num_points_total = int(cluster_sizes.sum())
        coord_dtype = self.itype if num_points_total else self.ftype
        np_voxels = np.empty((num_points_total, 2), dtype=coord_dtype)
        np_features = np.empty((num_points_total, 2), dtype=self.ftype)
        np_features[:, 1] = np.repeat(
            np.arange(num_clusters, dtype=self.ftype), cluster_sizes
        )

        # Loop over clusters to unpack their coordinates and values.
        point_offset = 0
        for cluster, num_points in zip(clusters, cluster_sizes):
            if num_points > 0:
                x = np.empty(num_points, dtype=np.int32)
                y = np.empty(num_points, dtype=np.int32)
                value = np.empty(num_points, dtype=np.float32)
                larcv.as_flat_arrays(cluster, meta, x, y, value)
                point_end = point_offset + num_points
                np_voxels[point_offset:point_end, 0] = x
                np_voxels[point_offset:point_end, 1] = y
                np_features[point_offset:point_end, 0] = value
                point_offset = point_end

        # Evaluate shifts to apply to each index column
        index_shifts = np.max(np_features[:, -1], keepdims=True, initial=-1) + 1

        return TensorData(
            coords=np_voxels,
            features=np_features,
            meta=Meta.from_larcv(meta),
            remove_duplicates=True,
            index_shifts=index_shifts,
            index_cols=self.index_cols,
            sum_cols=self.sum_cols,
        )


class LArCVCluster3DParser(ParserBase):
    """Class that retrieves and parses a 3D cluster list.

    .. code-block. yaml

        schema:
          cluster_label:
            parser: cluster3d
            cluster_event: cluster3d_pcluster
            sparse_semantics_event: sparse3d_semantics
            sparse_value_event: sparse3d_pcluster
            clean_data: true
            label_le: false
            break_clusters: false
            particle_info:
              particle_event: particle_pcluster
              particle_mpv_event: particle_mpv
              neutrino_event: neutrino_mpv
              label_le: false
              type_include_secondary: false
              type_include_mpr: false
              primary_include_mpr: true
    """

    # Name of the parser (as specified in the configuration)
    name = "cluster3d"

    # Type of object(s) returned by the parser

    def __init__(
        self,
        dtype: str,
        particle_event: Any | None = None,
        add_particle_info: bool = False,
        particle_info: Mapping[str, Any] | bool | None = None,
        clean_data: bool = False,
        label_le: bool | None = None,
        type_include_mpr: bool | None = None,
        type_include_secondary: bool | None = None,
        primary_include_mpr: bool | None = None,
        break_clusters: bool = False,
        break_eps: float = 1.1,
        break_metric: str = "chebyshev",
        shape_precedence: np.ndarray | list[int] | tuple[int, ...] = SHAPE_PREC,
        **kwargs: Any,
    ) -> None:
        """Initialize the parser.

        Parameters
        ----------
        particle_event : larcv.EventParticle, optional
            Legacy top-level particle input. Prefer ``particle_info``.
        add_particle_info : bool, default False
            Legacy switch which enables the particle table. Prefer
            ``particle_info``.
        particle_info : mapping or bool, optional
            Nested particle-input configuration. A mapping enables the named
            particle table and may contain ``particle_event``,
            ``particle_mpv_event``, ``neutrino_event``, ``type_include_mpr``,
            ``type_include_secondary``, ``primary_include_mpr`` and ``label_le``.
            ``None`` produces association-only cluster labels.
        clean_data : bool, default False
            If `True`, removes duplicate voxels
        label_le : bool, optional
            Legacy top-level form of ``particle_info.label_le``. Defaults to
            `False`.
        type_include_mpr : bool, optional
            Legacy top-level form of ``particle_info.type_include_mpr``.
            Defaults to `True`.
        type_include_secondary : bool, optional
            Legacy top-level form of ``particle_info.type_include_secondary``.
            Defaults to `True`.
        primary_include_mpr : bool, optional
            Legacy top-level form of ``particle_info.primary_include_mpr``.
            Defaults to `True`.
        break_clusters : bool, default False
            If `True`, runs DBSCAN on each cluster, assigns different cluster
            IDs to different fragments of the broken-down cluster
        break_eps : float, default 1.1
            Distance scale used in the break up procedure
        break_metric : str, default 'chebyshev'
            Distance metric used in the break up produce
        shape_precedence: list, default SHAPE_PREC
             Array of classes in the reference array, ordered by precedence
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Normalize the new nested particle configuration and legacy aliases
        if isinstance(particle_info, Mapping):
            particle_cfg = dict(particle_info)
            if "particle_event" in particle_cfg:
                if particle_event is not None:
                    raise ValueError(
                        "Particle input `particle_event` was specified twice."
                    )
                particle_event = particle_cfg.pop("particle_event")

            # Particle and neutrino products are loaded by the parser base
            for key in ("particle_mpv_event", "neutrino_event"):
                if key not in particle_cfg:
                    continue
                if key in kwargs:
                    raise ValueError(f"Particle input `{key}` was specified twice.")
                kwargs[key] = particle_cfg.pop(key)

            # Label-selection options belong to the particle-table configuration
            particle_options = {
                "label_le": label_le,
                "type_include_mpr": type_include_mpr,
                "type_include_secondary": type_include_secondary,
                "primary_include_mpr": primary_include_mpr,
            }
            for key, legacy_value in particle_options.items():
                if key not in particle_cfg:
                    continue
                if legacy_value is not None:
                    raise ValueError(f"Particle option `{key}` was specified twice.")
                particle_options[key] = particle_cfg.pop(key)
            type_include_mpr = particle_options["type_include_mpr"]
            type_include_secondary = particle_options["type_include_secondary"]
            primary_include_mpr = particle_options["primary_include_mpr"]
            label_le = particle_options["label_le"]

            if particle_cfg:
                unknown = ", ".join(sorted(particle_cfg))
                raise ValueError(f"Unknown particle information option(s): {unknown}.")
            include_particle_info = True
        elif particle_info is not None:
            include_particle_info = bool(particle_info)
        else:
            include_particle_info = add_particle_info

        # Initialize the parent class
        super().__init__(dtype, particle_event=particle_event, **kwargs)

        # Store the revelant attributes
        self.include_particle_info = include_particle_info
        self.clean_data = clean_data
        self.label_le = False if label_le is None else label_le
        self.type_include_mpr = True if type_include_mpr is None else type_include_mpr
        self.type_include_secondary = (
            True if type_include_secondary is None else type_include_secondary
        )
        self.primary_include_mpr = (
            True if primary_include_mpr is None else primary_include_mpr
        )
        self.shape_precedence = np.asarray(shape_precedence)
        if -1 not in self.shape_precedence:
            self.shape_precedence = np.append(self.shape_precedence, -1)

        # Initialize DBSCAN if the clusters are to be broken up
        self.break_clusters = break_clusters
        self.break_eps = break_eps
        self.break_metric_id = METRICS[break_metric]

        # Initialize the sparse parser
        self.sparse_parser = LArCVSparse3DParser(dtype, sparse_event="dummy")

        # If particle information is included, check that it is provided
        if self.include_particle_info and particle_event is None:
            raise ValueError(
                "If particle information is requested, `particle_event` "
                "must be provided."
            )

        # Define duplicate reduction for the temporary compact feature table.
        self.sum_cols = np.array([0], dtype=np.int64)

    def __call__(self, trees: dict[str, Any]) -> ClusterLabelData:
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object

        Returns
        -------
        ClusterLabelData
            Compact voxel associations and optional particle information.
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self,
        cluster_event: Any,
        particle_event: Any | None = None,
        particle_mpv_event: Any | None = None,
        neutrino_event: Any | None = None,
        sparse_semantics_event: Any | None = None,
        sparse_value_event: Any | None = None,
    ) -> ClusterLabelData:
        """Parse a list of 3D clusters into a structured cluster label.

        Parameters
        ----------
        cluster_event : larcv.EventClusterVoxel3D
            Event which contains the 3D clusters
        particle_event : larcv.EventParticle, optional
            List of true particle information used to populate the particle
            table.
        particle_mpv_event : larcv.EventParticle, optional
            List of true particle information for MPV particles only. If
            provided, it is used to determine which particles are MPV
        neutrino_event : larcv.EventNeutrino, optional
            List of true neutrino information. If provided, it is used
            to determine which particles are MPV
        sparse_semantics_event : larcv.EventSparseTensor3D, optional
            Semantics of each of the voxels in the image. If provided,
            overrides the order of precedence used in combining clusters
            which share voxels.
        sparse_value_event : larcv.EventSparseTensor3D, optional
            Value of each of the voxels in the image. If provided,
            Overrides the value provided by the list of 3D clusters itself.

        Returns
        -------
        ClusterLabelData
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                (N, 2) array of [voxel value, cluster ID], with an optional
                third particle-table index column.
            particles : dict[str, np.ndarray], optional
                Named particle-level arrays. These are omitted when
                ``particle_info`` is ``None``.
            meta : Meta
                Metadata of the parsed image
        """
        # Check that the semantics tensor is provided if `clean_data` is True
        if self.clean_data and sparse_semantics_event is None:
            raise ValueError(
                "A semantics tensor is required when `clean_data` is True."
            )

        # Get the cluster-wise information first
        meta = cluster_event.meta()
        num_clusters = cluster_event.as_vector().size()
        particle_table = None
        if self.include_particle_info:
            # Check that that particle objects are of the expected length
            if particle_event is None:  # pragma: no cover - validated in __init__
                raise RuntimeError("Particle input was not loaded.")
            num_particles = particle_event.size()
            if num_particles not in (num_clusters, num_clusters - 1):
                raise ValueError(
                    f"The number of particles ({num_particles}) must be "
                    f"aligned with the number of clusters ({num_clusters}). "
                    "There can be one more catch-all cluster at the end."
                )

            # Load up the particle list
            particles = list(particle_event.as_vector())

            # Fetch the variables missing from the larcv objects
            (
                _,
                group_ids,
                ancestor_ids,
                interaction_ids,
                nu_ids,
                group_primaries,
                inter_primaries,
                types,
            ) = process_particle_event(
                particle_event,
                particle_mpv_event,
                neutrino_event,
                label_le=self.label_le,
            )

            # Resolve ancestor track IDs to local particle-table indexes.
            track_id_to_index = {
                int(p.track_id()): index for index, p in enumerate(particles)
            }
            ancestor_indexes = np.asarray(
                [track_id_to_index.get(int(track_id), -1) for track_id in ancestor_ids],
                dtype=np.int64,
            )

            # Build a named particle table. Relationships are stored as local
            # indexes, while physical attributes retain their natural dtype.
            anc_pos = image_coordinates_batch(
                meta,
                particles,
                dtype=self.ftype,
                position_attr="ancestor_position",
            )
            particle_table = {
                "particle": np.asarray([p.id() for p in particles], dtype=np.int64),
                "group": np.asarray(group_ids, dtype=np.int64),
                "ancestor": ancestor_indexes,
                "interaction": np.asarray(interaction_ids, dtype=np.int64),
                "nu": np.asarray(nu_ids, dtype=np.int64),
                "pid": np.asarray(types, dtype=np.int64),
                "group_primary": np.asarray(group_primaries, dtype=np.int64),
                "interaction_primary": np.asarray(inter_primaries, dtype=np.int64),
                "vertex": np.asarray(anc_pos, dtype=self.ftype),
                "momentum": np.asarray([p.p() for p in particles], dtype=self.ftype),
                "energy_init": np.asarray(
                    [p.energy_init() for p in particles], dtype=self.ftype
                ),
                "shape": np.asarray([p.shape() for p in particles], dtype=np.int64),
            }

            # If requested, give invalid labels to a subset of particles
            if not self.type_include_secondary:
                secondary_mask = np.where(particle_table["interaction_primary"] < 1)[0]
                particle_table["pid"][secondary_mask] = -1

            if not self.type_include_mpr or not self.primary_include_mpr:
                mpr_mask = np.where(particle_table["nu"] < 0)[0]
                if not self.type_include_mpr:
                    particle_table["pid"][mpr_mask] = -1
                if not self.primary_include_mpr:
                    particle_table["interaction_primary"][mpr_mask] = -1

        # Allocate the output tensors once. Events may contain thousands of
        # clusters, so building one array per feature per cluster and then
        # concatenating them creates substantial Python and allocator overhead.
        clusters = list(cluster_event.as_vector())
        cluster_sizes = np.fromiter(
            (cluster.as_vector().size() for cluster in clusters),
            dtype=np.int64,
            count=num_clusters,
        )
        num_points_total = int(cluster_sizes.sum())
        np_voxels = np.empty((num_points_total, 3), dtype=self.itype)
        num_features = 3 if particle_table is not None else 2
        np_features = np.empty((num_points_total, num_features), dtype=self.ftype)
        np_features[:, 1] = np.repeat(
            np.arange(num_clusters, dtype=self.ftype), cluster_sizes
        )
        if particle_table is not None:
            cluster_particles = np.full(num_clusters, -1, dtype=self.ftype)
            particle_count = len(particle_table["particle"])
            cluster_particles[:particle_count] = np.arange(particle_count)
            np_features[:, 2] = np.repeat(cluster_particles, cluster_sizes)

        # Loop over clusters to unpack their coordinates and values.
        id_offset = 0
        point_offset = 0
        for i, (cluster, num_points) in enumerate(zip(clusters, cluster_sizes)):
            if num_points > 0:
                # Get the position and pixel value from EventSparseTensor3D
                x = np.empty(num_points, dtype=np.int32)
                y = np.empty(num_points, dtype=np.int32)
                z = np.empty(num_points, dtype=np.int32)
                value = np.empty(num_points, dtype=np.float32)
                larcv.as_flat_arrays(cluster, meta, x, y, z, value)
                point_end = point_offset + num_points
                voxels = np_voxels[point_offset:point_end]
                voxels[:, 0] = x
                voxels[:, 1] = y
                voxels[:, 2] = z
                np_features[point_offset:point_end, 0] = value

                # If requested, break cluster into detached pieces
                if self.break_clusters:
                    frag_labels = dbscan(
                        voxels,
                        eps=self.break_eps,
                        min_samples=1,
                        metric_id=self.break_metric_id,
                    )
                    np_features[point_offset:point_end, 1] = id_offset + frag_labels
                    id_offset += np.max(frag_labels, initial=-1) + 1

                if (
                    particle_table is not None
                    and i < len(particle_table["shape"])
                    and particle_table["shape"][i] >= LOWES_SHP + self.label_le
                ):
                    np_features[point_offset:point_end, 1:] = -1

                point_offset = point_end

        # If requested, remove duplicate voxels (cluster overlaps) and
        # match the semantics to those of the provided reference
        if len(np_voxels) and (
            (sparse_semantics_event is not None) or (sparse_value_event is not None)
        ):
            if not self.clean_data:
                warn(
                    "You must set `clean_data` to `True` if you specify a "
                    "sparse tensor in `Cluster3DParser`."
                )
                self.clean_data = True

            # Build a temporary semantic column solely for precedence-based
            # overlap cleaning; it is not retained in the cluster-label product.
            if sparse_semantics_event is None:
                raise ValueError(
                    "A semantics tensor is required to clean cluster labels."
                )
            tensor_seg = self.sparse_parser.process(sparse_semantics_event)
            semantic = np.full((len(np_features), 1), -1, dtype=self.ftype)
            if particle_table is not None:
                particle_index = np_features[:, 2].astype(np.int64)
                valid = particle_index >= 0
                semantic[valid, 0] = particle_table["shape"][particle_index[valid]]
            clean_features = np.concatenate((np_features, semantic), axis=1)
            np_voxels, clean_features = clean_sparse_data(
                np_voxels,
                clean_features,
                tensor_seg.coords,
                precedence=self.shape_precedence,
                sum_cols=self.sum_cols,
            )
            np_features = clean_features[:, :-1]

            # Invalidate associations for semantic classes that are not retained.
            shape_mask = tensor_seg.features[:, -1] >= LOWES_SHP + self.label_le
            np_features[shape_mask, 1:] = -1

            # If a value tree is provided, override value colum
            if sparse_value_event is not None:
                tensor_val = self.sparse_parser.process(sparse_value_event)
                np_features[:, 0] = tensor_val.features[:, 0]

        return ClusterLabelData(
            coords=np_voxels,
            features=np_features,
            particles=particle_table,
            meta=Meta.from_larcv(meta),
            remove_duplicates=True,
            sum_cols=self.sum_cols,
            precedence=self.shape_precedence if particle_table is not None else None,
        )


class LArCVCluster3DAggregateParser(LArCVCluster3DParser):
    """Identical to :class:`Cluster3DParser`, but aggregates charge information
    from multiple value sources.
    """

    # Name of the parser (as specified in the configuration)
    name = "cluster3d_aggr"

    def __init__(
        self,
        dtype: str,
        sparse_value_event_list: list[str],
        value_aggr: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the parser.

        Parameters
        ----------
        sparse_value_event_list : List[larcv.EventSparseTensor3D]
            List of sparse tensors used to compute the aggregated charge
        value_aggr : str
            Value aggregation function to apply ('sum', 'mean', 'max', etc.)
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(
            dtype, sparse_value_event_list=sparse_value_event_list, **kwargs
        )

        # Initialize the sparse parser which computes the rescaled charge
        self.sparse_aggr_parser = LArCVSparse3DAggregateParser(
            dtype, sparse_event_list=sparse_value_event_list, aggr=value_aggr
        )

    def __call__(self, trees: dict[str, Any]) -> ClusterLabelData:
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object

        Returns
        -------
        ClusterLabelData
            Cluster labels with values aggregated from sparse inputs.
        """
        return self.process_aggr(**self.get_input_data(trees))

    def process_aggr(
        self, sparse_value_event_list: list[Any], **kwargs: Any
    ) -> ClusterLabelData:
        """Parse a list of 3D clusters into a single tensor and fetch the
        value column by aggregating multiple tensor features.

        Parameters
        ----------
        sparse_value_event_list : List[larcv.EventSparseTensor3D]
            List of sparse value tensors
        **kwargs : dict, optional
            Extra data products to pass to the parent Cluster3DParser

        Returns
        -------
        ClusterLabelData
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                Compact [voxel value, cluster ID, particle index?] features.
            particles : dict[str, np.ndarray], optional
                Named particle-level arrays inherited from the base parser.
            meta : Meta
                Metadata of the parsed image
        """
        # Process the input using the main parser
        tensor = self.process(**kwargs)

        # Modify the value column using the aggregate tensor values
        tensor_val = self.sparse_aggr_parser.process_aggr(sparse_value_event_list)
        tensor.features[:, 0] = tensor_val.features[:, 0]

        return tensor


class LArCVCluster3DChargeRescaledParser(LArCVCluster3DParser):
    """Identical to :class:`Cluster3DParser`, but computes rescaled charges
    on the fly.
    """

    # Name of the parser (as specified in the configuration)
    name = "cluster3d_rescale_charge"

    def __init__(
        self,
        dtype: str,
        sparse_value_event_list: list[str],
        collection_only: bool = False,
        collection_id: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize the parser.

        Parameters
        ----------
        sparse_value_event_list : List[larcv.EventSparseTensor3D]
            (7) List of sparse tensors used to compute the rescaled charge
            - Charge value of each of the contributing planes (3)
            - Index of the plane hit contributing to the space point (3)
            - Semantic labels (1)
        collection_only : bool, default False
            If True, only uses the collection plane charge
        collection_id : int, default 2
            Index of the collection plane
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(
            dtype, sparse_value_event_list=sparse_value_event_list, **kwargs
        )

        # Initialize the sparse parser which computes the rescaled charge
        self.sparse_rescale_parser = LArCVSparse3DChargeRescaledParser(
            dtype,
            sparse_event_list=sparse_value_event_list,
            collection_only=collection_only,
            collection_id=collection_id,
        )

    def __call__(self, trees: dict[str, Any]) -> ClusterLabelData:
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object

        Returns
        -------
        ClusterLabelData
            Cluster labels with charge-rescaled voxel values.
        """
        return self.process_rescale(**self.get_input_data(trees))

    def process_rescale(
        self, sparse_value_event_list: list[Any], **kwargs: Any
    ) -> ClusterLabelData:
        """Parse a list of 3D clusters into a single tensor and reset
        the value column by rescaling the charge coming from 3 wire planes.

        Parameters
        ----------
        sparse_value_event_list : List[larcv.EventSparseTensor3D]
            (7) List of sparse tensors used to compute the rescaled charge
            - Charge value of each of the contributing planes (3)
            - Index of the plane hit contributing to the space point (3)
            - Semantic labels (1)
        **kwargs : dict, optional
            Extra data products to pass to the parent Cluster3DParser

        Returns
        -------
        ClusterLabelData
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                Compact [voxel value, cluster ID, particle index?] features.
            particles : dict[str, np.ndarray], optional
                Named particle-level arrays inherited from the base parser.
            meta : Meta
                Metadata of the parsed image
        """
        # Process the input using the main parser
        tensor = self.process(**kwargs)

        # Modify the value column using the charge rescaled on the fly
        tensor_val = self.sparse_rescale_parser.process_rescale(sparse_value_event_list)
        tensor.features[:, 0] = tensor_val.features[:, 0]

        return tensor
