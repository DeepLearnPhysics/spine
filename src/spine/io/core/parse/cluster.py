"""Module that contains all parsers related to LArCV cluster data.

Contains the following parsers:
- :class:`Cluster2DParser`
- :class:`Cluster3DParser`
- :class:`Cluster3DAggregateParser`
- :class:`Cluster3DChargeRescaledParser`
"""

from collections import OrderedDict
from warnings import warn

import numpy as np

from spine.data import Meta
from spine.math.cluster import dbscan
from spine.math.distance import METRICS
from spine.utils.conditional import larcv
from spine.utils.globals import (
    CLUST_COL,
    DELTA_SHP,
    GROUP_COL,
    INTER_COL,
    NU_COL,
    PART_COL,
    SHAPE_COL,
    SHAPE_PREC,
    VALUE_COL,
)
from spine.utils.particles import process_particle_event
from spine.utils.ppn import image_coordinates

from .base import ParserBase
from .clean_data import clean_sparse_data
from .data import ParserTensor
from .sparse import (
    Sparse3DAggregateParser,
    Sparse3DChargeRescaledParser,
    Sparse3DParser,
)

__all__ = [
    "Cluster2DParser",
    "Cluster3DParser",
    "Cluster3DAggregateParser",
    "Cluster3DChargeRescaledParser",
]


class Cluster2DParser(ParserBase):
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
    returns = "tensor"

    def __init__(self, dtype, cluster_event, projection_id):
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

        # Store the revelant attributes
        self.projection_id = projection_id

        # Define the overlay strategy parameters
        self.index_cols = np.array([CLUST_COL])
        self.sum_cols = np.array([VALUE_COL])

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(self, cluster_event):
        """Converts a 2D clusters tensor into a single tensor.

        Parameters
        ----------
        cluster_event : larcv.EventClusterPixel2D
            Event which contains the 2D clusters

        Returns
        -------
        ParserTensor
            coords : np.ndarray
                (N, 2) array of [x, y] coordinates
            features : np.ndarray
                (N, 2) array of [pixel value, cluster ID]
            meta : Meta
                Metadata of the parsed image
        """
        # Get the cluster from the appropriate projection
        cluster_event_p = cluster_event.cluster_pixel_2d(self.projection_id)

        # Loop over clusters, store information
        meta = cluster_event_p.meta()
        num_clusters = cluster_event_p.size()
        clusters_voxels, clusters_features = [], []
        for i in range(num_clusters):
            cluster = cluster_event_p.as_vector()[i]
            num_points = cluster.as_vector().size()
            if num_points > 0:
                x = np.empty(num_points, dtype=np.int32)
                y = np.empty(num_points, dtype=np.int32)
                value = np.empty(num_points, dtype=np.float32)
                larcv.as_flat_arrays(cluster, meta, x, y, value)
                pixels = np.stack([x, y], axis=1).astype(self.itype)
                value = value.astype(self.ftype)
                cluster_id = np.full(num_points, i, dtype=self.ftype)
                clusters_voxels.append(pixels)
                clusters_features.append(np.column_stack([value, cluster_id]))

        # Concatenate clusters
        if not clusters_voxels:
            np_voxels = np.empty((0, 2), dtype=self.ftype)
            np_features = np.empty((0, 2), dtype=self.ftype)
        else:
            np_voxels = np.concatenate(clusters_voxels, axis=0)
            np_features = np.concatenate(clusters_features, axis=0)

        # Evaluate shifts to apply to each index column
        index_shifts = np.max(np_features[:, -1], keepdims=True, initial=-1) + 1

        return ParserTensor(
            coords=np_voxels,
            feats=np_features,
            meta=Meta.from_larcv(meta),
            remove_duplicates=True,
            index_shifts=index_shifts,
            index_cols=self.index_cols,
            sum_cols=self.sum_cols,
        )


class Cluster3DParser(ParserBase):
    """Class that retrieves and parses a 3D cluster list.

    .. code-block. yaml

        schema:
          cluster_label:
            parser: cluster3d
            cluster_event: cluster3d_pcluster
            particle_event: particle_pcluster
            particle_mpv_event: particle_mpv
            neutrino_event: neutrino_mpv
            sparse_semantics_event: sparse3d_semantics
            sparse_value_event: sparse3d_pcluster
            add_particle_info: true
            clean_data: true
            type_include_mpr: false
            type_include_secondary: false
            primary_include_mpr: true
            break_clusters: false
    """

    # Name of the parser (as specified in the configuration)
    name = "cluster3d"

    # Type of object(s) returned by the parser
    returns = "tensor"

    def __init__(
        self,
        dtype,
        particle_event=None,
        add_particle_info=False,
        clean_data=False,
        type_include_mpr=True,
        type_include_secondary=True,
        primary_include_mpr=True,
        break_clusters=False,
        break_eps=1.1,
        break_metric="chebyshev",
        shape_precedence=SHAPE_PREC,
        **kwargs,
    ):
        """Initialize the parser.

        Parameters
        ----------
        particle_event : larcv.EventParticle, optional
            List of true particle information. If prodided, allows to fetch
            more information about each of the pixels in the image
        add_particle_info : bool, default False
            If `True`, adds truth information from the true particle list
        clean_data : bool, default False
            If `True`, removes duplicate voxels
        type_include_mpr : bool, default False
            If `False`, sets all PID labels to -1 for MPR particles
        type_include_secondary : bool, default False
            If `False`, sets all PID labels to -1 for secondary particles
        primary_include_mpr : bool, default False
            If `False`, sets all primary labels to -1 for MPR particles
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
        # Initialize the parent class
        super().__init__(dtype, particle_event=particle_event, **kwargs)

        # Store the revelant attributes
        self.add_particle_info = add_particle_info
        self.clean_data = clean_data
        self.type_include_mpr = type_include_mpr
        self.type_include_secondary = type_include_secondary
        self.primary_include_mpr = primary_include_mpr
        self.shape_precedence = shape_precedence

        # Intialize DBSCAN if the clusters are to be broken up
        self.break_clusters = break_clusters
        self.break_eps = break_eps
        self.break_metric_id = METRICS[break_metric]

        # Intialize the sparse and particle parsers
        self.sparse_parser = Sparse3DParser(dtype, sparse_event="dummy")

        # If particle information is to be incldued, check that it is provided
        if self.add_particle_info:
            # Must provide particle event to build particle info
            assert particle_event is not None, (
                "If `add_particle_info` is `True`, must provide the "
                "`particle_event` argument."
            )

        # Define the overlay strategy parameters
        self.sum_cols = np.array([VALUE_COL])
        if not self.add_particle_info:
            self.index_cols = np.array([CLUST_COL])
            self.prec_col = None
        else:
            self.index_cols = np.array(
                [CLUST_COL, PART_COL, GROUP_COL, INTER_COL, NU_COL]
            )
            self.prec_col = SHAPE_COL

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(
        self,
        cluster_event,
        particle_event=None,
        particle_mpv_event=None,
        neutrino_event=None,
        sparse_semantics_event=None,
        sparse_value_event=None,
    ):
        """Parse a list of 3D clusters into a single tensor.

        Parameters
        ----------
        cluster_event : larcv.EventClusterVoxel3D
            Event which contains the 3D clusters
        particle_event : larcv.EventParticle, optional
            List of true particle information. If prodided, allows to fetch
            more information about each of the pixels in the image
        particle_mpv_event : larcv.EventParticle, optional
            List of true particle information for MPV particles only. If
            provided, it is used to determine which particles are MPV
        particle_mpv_event: larcv.EventNeutrino, optional
            List of true neutrino information. If provided, it is used
            to determine which particles are MPV
        sparse_semantics_event : larcv.EventSparseTensor3D, optional
            Semantics of each of the voxels in the image. If provided,
            overrides the order of precedence used in combining clusters
            which share voxels.
        sparse_value_event : larcv.EventSparseTensor3D, optional
            Value of each of the voxels in the image. If provided,
            overrides the value provided byt eh list of 3D clusters itself

        Returns
        -------
        ParserTensor
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                (N, 2/14) array of features, minimally [voxel value, cluster ID].
                If `add_particle_info` is `True`, the additonal columns are
                [particle ID, group ID, interaction ID, neutrino ID, particle type,
                group primary bool, interaction primary bool, vertex x, vertex y,
                vertex z, momentum, semantic type]
            meta : Meta
                Metadata of the parsed image
        """
        # Get the cluster-wise information first
        meta = cluster_event.meta()
        num_clusters = cluster_event.as_vector().size()
        labels = OrderedDict()
        labels["cluster"] = np.arange(num_clusters)
        num_particles = num_clusters
        if self.add_particle_info:
            # Check that that particle objects are of the expected length
            num_particles = particle_event.size()
            assert num_particles in (num_clusters, num_clusters - 1), (
                f"The number of particles ({num_particles}) must be "
                f"aligned with the number of clusters ({num_clusters}). "
                f"There can me one more catch-all cluster at the end."
            )

            # Load up the particle list
            particles = list(particle_event.as_vector())

            # Fetch the variables missing from the larcv objects
            (inter_ids, nu_ids, group_primaries, inter_primaries, types) = (
                process_particle_event(
                    particle_event, particle_mpv_event, neutrino_event
                )
            )

            # Store the cluster ID information
            labels["cluster"] = [p.id() for p in particles]
            labels["part"] = [p.id() for p in particles]
            labels["group"] = [p.group_id() for p in particles]
            labels["inter"] = inter_ids
            labels["nu"] = nu_ids

            # Store the type/primary status
            labels["type"] = types
            labels["pgroup"] = group_primaries
            labels["pinter"] = inter_primaries

            # Store the vertex and momentum
            anc_pos = np.empty((len(particles), 3), dtype=self.ftype)
            for i, p in enumerate(particles):
                anc_pos[i] = image_coordinates(meta, p.ancestor_position())
            labels["vtx_x"] = anc_pos[:, 0]
            labels["vtx_y"] = anc_pos[:, 1]
            labels["vtx_z"] = anc_pos[:, 2]
            labels["p"] = [p.p() for p in particles]

            # Store the shape last (consistent with semantics tensor)
            labels["shape"] = [p.shape() for p in particles]

            # If requested, give invalid labels to a subset of particles
            if not self.type_include_secondary:
                secondary_mask = np.where(np.array(labels["pinter"]) < 1)[0]
                labels["type"] = np.asarray(labels["type"])
                labels["type"][secondary_mask] = -1

            if not self.type_include_mpr or not self.primary_include_mpr:
                mpr_mask = np.where(np.array(labels["nu"]) < 0)[0]
                if not self.type_include_mpr:
                    labels["type"] = np.asarray(labels["type"])
                    labels["type"][mpr_mask] = -1
                if not self.primary_include_mpr:
                    labels["pinter"] = np.asarray(labels["pinter"])
                    labels["pinter"][mpr_mask] = -1

            # Count the number of neutrinos
            if neutrino_event is not None:
                num_neutrinos = len(neutrino_event.as_vector())
            else:
                num_neutrinos = np.max(nu_ids, initial=-1) + 1

        # Loop over clusters, store information
        clusters_voxels, clusters_features = [], []
        id_offset = 0
        for i in range(num_clusters):
            cluster = cluster_event.as_vector()[i]
            num_points = cluster.as_vector().size()
            if num_points > 0:
                # Get the position and pixel value from EventSparseTensor3D
                x = np.empty(num_points, dtype=np.int32)
                y = np.empty(num_points, dtype=np.int32)
                z = np.empty(num_points, dtype=np.int32)
                value = np.empty(num_points, dtype=np.float32)
                larcv.as_flat_arrays(cluster, meta, x, y, z, value)
                voxels = np.stack([x, y, z], axis=1).astype(self.itype)
                value = value.astype(self.ftype)
                clusters_voxels.append(voxels)

                # Append the cluster-wise information
                features = [value]
                for l in labels.values():
                    val = l[i] if i < num_particles else -1
                    features.append(np.full(num_points, val, dtype=self.ftype))

                # If requested, break cluster into detached pieces
                if self.break_clusters:
                    frag_labels = dbscan(
                        voxels,
                        eps=self.break_eps,
                        min_samples=1,
                        metric_id=self.break_metric_id,
                    )
                    features[1] = id_offset + frag_labels
                    id_offset += np.max(frag_labels, initial=-1) + 1

                clusters_features.append(np.column_stack(features))

        # If there are no non-empty clusters, return. Concatenate otherwise
        if not clusters_voxels:
            np_voxels = np.empty((0, 3), dtype=self.itype)
            np_features = np.empty((0, len(labels) + 1), dtype=self.ftype)
        else:
            np_voxels = np.concatenate(clusters_voxels, axis=0)
            np_features = np.concatenate(clusters_features, axis=0)

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

            # Extract voxels and features
            assert self.add_particle_info, (
                "Need to add particle info to fetch particle "
                "semantics for each voxel."
            )
            assert (
                sparse_semantics_event is not None
            ), "Need to provide a semantics tensor to clean up output."
            tensor_seg = self.sparse_parser.process(sparse_semantics_event)
            np_voxels, np_features = clean_sparse_data(
                np_voxels,
                np_features,
                tensor_seg.coords,
                precedence=self.shape_precedence,
                sum_cols=self.sum_cols - VALUE_COL,
            )

            # Match the semantic column to the reference tensor
            np_features[:, -1] = tensor_seg.features[:, -1]

            # Set all cluster labels to -1 if semantic class is LE or ghost
            shape_mask = tensor_seg.features[:, -1] > DELTA_SHP
            np_features[shape_mask, 1:-1] = -1

            # If a value tree is provided, override value colum
            if sparse_value_event:
                tensor_val = self.sparse_parser.process(sparse_value_event)
                np_features[:, 0] = tensor_val.features[:, 0]

        # Evaluate shifts to apply to each index column
        clust_shift = np.max(np_features[:, 1], initial=-1) + 1
        if not self.add_particle_info:
            index_shifts = np.array([clust_shift])
        else:
            cs, ps, ns = clust_shift, num_particles, num_neutrinos
            index_shifts = np.array([cs, ps, ps, ps, ns])

        return ParserTensor(
            coords=np_voxels,
            features=np_features,
            meta=Meta.from_larcv(meta),
            remove_duplicates=True,
            index_shifts=index_shifts,
            index_cols=self.index_cols,
            sum_cols=self.sum_cols,
            prec_col=self.prec_col,
            precedence=self.shape_precedence,
        )


class Cluster3DAggregateParser(Cluster3DParser):
    """Identical to :class:`Cluster3DParser`, but aggregates charge information
    from multiple value sources.
    """

    # Name of the parser (as specified in the configuration)
    name = "cluster3d_aggr"

    def __init__(self, dtype, sparse_value_event_list, value_aggr, **kwargs):
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
        self.sparse_aggr_parser = Sparse3DAggregateParser(
            dtype, sparse_event_list=sparse_value_event_list, aggr=value_aggr
        )

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process_aggr(**self.get_input_data(trees))

    def process_aggr(self, sparse_value_event_list, **kwargs):
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
        ParserTensor
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                (N, 2/14) array of features, minimally [voxel value, cluster ID].
                If `add_particle_info` is `True`, the additonal columns are
                [group ID, interaction ID, neutrino ID, particle type,
                group primary bool, interaction primary bool, vertex x, vertex y,
                vertex z, momentum, semantic type, particle ID]
            meta : Meta
                Metadata of the parsed image
        """
        # Process the input using the main parser
        tensor = self.process(**kwargs)

        # Modify the value column using the aggregate tensor values
        tensor_val = self.sparse_aggr_parser.process_aggr(sparse_value_event_list)
        tensor.features[:, 0] = tensor_val.features[:, 0]

        return tensor


class Cluster3DChargeRescaledParser(Cluster3DParser):
    """Identical to :class:`Cluster3DParser`, but computes rescaled charges
    on the fly.
    """

    # Name of the parser (as specified in the configuration)
    name = "cluster3d_rescale_charge"

    def __init__(
        self,
        dtype,
        sparse_value_event_list,
        collection_only=False,
        collection_id=2,
        **kwargs,
    ):
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
        self.sparse_rescale_parser = Sparse3DChargeRescaledParser(
            dtype,
            sparse_event_list=sparse_value_event_list,
            collection_only=collection_only,
            collection_id=collection_id,
        )

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process_rescale(**self.get_input_data(trees))

    def process_rescale(self, sparse_value_event_list, **kwargs):
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
        ParserTensor
            coords : np.ndarray
                (N, 3) array of [x, y, z] coordinates
            features : np.ndarray
                (N, 2/14) array of features, minimally [voxel value, cluster ID].
                If `add_particle_info` is `True`, the additonal columns are
                [group ID, interaction ID, neutrino ID, particle type,
                group primary bool, interaction primary bool, vertex x, vertex y,
                vertex z, momentum, semantic type, particle ID]
            meta : Meta
                Metadata of the parsed image
        """
        # Process the input using the main parser
        tensor = self.process(**kwargs)

        # Modify the value column using the charge rescaled on the fly
        tensor_val = self.sparse_rescale_parser.process_rescale(sparse_value_event_list)
        tensor.features[:, 0] = tensor_val.features[:, 0]

        return tensor
