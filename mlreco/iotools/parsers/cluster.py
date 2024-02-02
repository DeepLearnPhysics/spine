"""Module that contains all parsers related to LArCV cluster data.

Contains the following parsers:
- :class:`Cluster2DParser`
- :class:`Cluster3DParser`
"""

import numpy as np
from collections import OrderedDict
from larcv import larcv
from sklearn.cluster import DBSCAN

from .parser import Parser
from .sparse import Sparse3DParser
from .particles import ParticleParser
from .clean_data import clean_sparse_data

from mlreco.utils.globals import SHAPE_COL, DELTA_SHP, UNKWN_SHP
from mlreco.utils.data_structures import Meta
from mlreco.utils.unwrap import Unwrapper
from mlreco.utils.particles import (get_interaction_ids, get_nu_ids,
                                    get_particle_ids, get_shower_primary_ids,
                                    get_group_primary_ids)

__all__ = ['Cluster2DParser', 'Cluster3DParser',
           'Cluster3DChargeRescaledParser', 'Cluster3DMultiModuleParser']


class Cluster2DParser(Parser):
    """Class that retrieves and parses a 2D cluster list.

    .. code-block. yaml

        schema:
          cluster_label:
            parser: parse_cluster2d
            cluster_event: cluster2d_pcluster
    """
    name = 'parse_cluster2d'
    result = Unwrapper.Rule(method='tensor',
                            default=np.empty((0, 1 + 2 + 2), dtype=np.float32))

    def process(self, cluster_event):
        """Converts a 2D clusters tensor into a single tensor.

        Parameters
        ----------
        cluster_event : larcv.EventClusterPixel2D
            Event which contains the 2D clusters

        Returns
        -------
        np_voxels : np.ndarray
            (N, 2) array of [x, y] coordinates
        np_features : np.ndarray
            (N, 2) array of [pixel value, cluster ID]
        meta : Meta
            Metadata of the parsed image
        """
        # Loop over clusters, store information
        cluster_event = cluster_event.as_vector().front()
        meta = cluster_event.meta()
        num_clusters = cluster_event.size()
        clusters_voxels, clusters_features = [], []
        for i in range(num_clusters):
            cluster = cluster_event.as_vector()[i]
            num_points = cluster.as_vector().size()
            if num_points > 0:
                x = np.empty(num_points, dtype=np.int32)
                y = np.empty(num_points, dtype=np.int32)
                value = np.empty(num_points, dtype=np.float32)
                larcv.as_flat_arrays(cluster, meta, x, y, value)
                cluster_id = np.full(num_points, i, dtype=np.float32)
                clusters_voxels.append(np.stack([x, y], axis=1))
                clusters_features.append(np.column_stack([value, cluster_id]))

        # If there are no non-empty clusters, return. Concatenate otherwise
        if not len(clusters_voxels):
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32),
                    Meta.from_larcv(meta))

        np_voxels   = np.concatenate(clusters_voxels, axis=0)
        np_features = np.concatenate(clusters_features, axis=0)

        return np_voxels, np_features, Meta.from_larcv(meta)


class Cluster3DParser(Parser):
    """Class that retrieves and parses a 3D cluster list.

    .. code-block. yaml

        schema:
          cluster_label:
            parser: parse_cluster3d
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
            min_size: -1
    """
    name = 'parse_cluster3d'
    result = Unwrapper.Rule(method='tensor', translate=True)

    def __init__(self, particle_event=None, add_particle_info=False,
                 clean_data=False, type_include_mpr=False,
                 type_include_secondary=False, primary_include_mpr=False,
                 break_clusters=False, min_size=-1, **kwargs):
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
        min_size : int, default -1
            Minimum cluster size to be parsed in the combined tensor
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(particle_event=particle_event, **kwargs)

        # Store the revelant attributes
        self.add_particle_info = add_particle_info
        self.clean_data = clean_data
        self.type_include_mpr = type_include_mpr
        self.type_include_secondary = type_include_secondary
        self.primary_include_mpr = primary_include_mpr
        self.break_clusters = break_clusters
        self.min_size = max(min_size, 1)

        # Intialize the sparse and particle parsers
        self.sparse_parser = Sparse3DParser(sparse_event='dummy')
        self.particle_parser = ParticleParser()

        # Initialize the DBSCAN algorithm, if needed
        if self.break_clusters:
            self.dbscan = DBSCAN(eps=1.1, min_samples=1, metric='chebyshev')

        # Do basic sanity checks
        if self.add_particle_info:
            assert particle_event is not None, (
                    "If `add_particle_info` is `True`, must provide the"
                    "`particle_event` argument")

        # Based on the parameters, define a default output
        if not self.add_particle_info:
            self.result.default = np.empty((0, 1 + 3 + 2),
                                           dtype=np.float32)
        else:
            self.result.default = np.empty((0, 1 + 3 + 12),
                                           dtype=np.float32)

    def process(self, cluster_event, particle_event=None,
                particle_mpv_event=None, neutrino_event=None,
                sparse_semantics_event=None, sparse_value_event=None):
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
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, 2/14) array of features, minimally [voxel value, cluster ID].
            If `add_particle_info` is `True`, the additonal columns are
            [particle ID, group ID, interaction ID, neutrino ID, particle type,
            shower primary bool, primary group bool, vertex x, vertex y,
            vertex z, momentum, semantic type]
        meta : Meta
            Metadata of the parsed image
        """
        # Get the cluster-wise information first
        meta = cluster_event.meta()
        num_clusters = cluster_event.as_vector().size()
        labels = OrderedDict()
        labels['cluster'] = np.arange(num_clusters)
        if self.add_particle_info:
            # Check that that particle objects are of the expected length
            num_particles = particle_event.size()
            assert (num_particles == num_clusters or
                    num_particles == num_clusters - 1), (
                    f"The number of particles ({num_particles}) must be "
                    f"aligned with the number of clusters ({num_clusters}). "
                    f"There can me one more catch-all cluster at the end.")

            # Load up the particle/meutrino objects as lists
            particles = list(particle_event.as_vector())
            particles_p = self.particle_parser.process(
                    particle_event, cluster_event)

            particles_mpv, neutrinos = None, None
            if particle_mpv_event is not None:
                particles_mpv = list(particle_mpv_event.as_vector())
            if neutrino_event is not None:
                neutrinos = list(neutrino_event.as_vector())

            # Store the cluster ID information
            labels['cluster']  = np.array([p.id() for p in particles])
            labels['particle'] = np.array([p.id() for p in particles])
            labels['group']    = np.array([p.group_id() for p in particles])
            labels['inter']    = get_interaction_ids(particles)
            labels['nu']       = get_nu_ids(particles, labels['inter'],
                                            particles_mpv, neutrinos)

            # Store the type/primary status
            labels['type']    = get_particle_ids(
                    particles, labels['nu'], self.type_include_mpr,
                    self.type_include_secondary)
            labels['pshower'] = get_shower_primary_ids(particles)
            labels['pgroup']  = get_group_primary_ids(
                    particles, labels['nu'], self.primary_include_mpr)

            # Store the vertex and momentum
            anc_pos = lambda x : getattr(x, 'ancestor_position')()
            labels['vtx_x'] = np.array([anc_pos(p).x() for p in particles_p])
            labels['vtx_y'] = np.array([anc_pos(p).y() for p in particles_p])
            labels['vtx_z'] = np.array([anc_pos(p).z() for p in particles_p])
            labels['p']     = np.array([p.p()/1e3 for p in particles]) # In GeV

            # Store the shape last (consistent with semantics tensor)
            labels['shape'] = np.array([p.shape() for p in particles])

        # Loop over clusters, store information
        clusters_voxels, clusters_features = [], []
        id_offset = 0
        for i in range(num_clusters):
            cluster = cluster_event.as_vector()[i]
            num_points = cluster.as_vector().size()
            if num_points >= self.min_size:
                # Get the position and pixel value from EventSparseTensor3D
                x = np.empty(num_points, dtype=np.int32)
                y = np.empty(num_points, dtype=np.int32)
                z = np.empty(num_points, dtype=np.int32)
                value = np.empty(num_points, dtype=np.float32)
                larcv.as_flat_arrays(cluster, meta, x, y, z, value)
                voxels = np.stack([x, y, z], axis=1)
                clusters_voxels.append(voxels)

                # Append the cluster-wise information
                features = [value]
                for k, l in labels.items():
                    if i < len(l):
                        value = l[i]
                    else:
                        value = -1 if k != 'shape' else UNKWN_SHP
                    features.append(
                            np.full(num_points, value, dtype=np.float32))

                # If requested, break cluster into detached pieces
                if self.break_clusters:
                    frag_labels = np.unique(self.dbscan.fit(voxels).labels_,
                                            return_inverse=True)[-1]
                    features[1] = id_offset + frag_labels
                    id_offset += max(frag_labels) + 1

                clusters_features.append(np.column_stack(features))

        # If there are no non-empty clusters, return. Concatenate otherwise
        if not len(clusters_voxels):
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0, len(labels) + 1), dtype=np.float32),
                    Meta.from_larcv(meta))

        np_voxels   = np.concatenate(clusters_voxels, axis=0)
        np_features = np.concatenate(clusters_features, axis=0)

        # If requested, remove duplicate voxels (cluster overlaps) and
        # match the semantics to those of the provided reference
        if ((sparse_semantics_event is not None) or
            (sparse_value_event is not None)):
            # TODO: use proper logging
            if not self.clean_data:
                from warnings import warn
                warn("You must set `clean_data` to `True` if you specify a"
                     "sparse tensor in parse_cluster3d")
                self.clean_data = True

            # Extract voxels and features
            assert self.add_particle_info, (
                    "Need to add particle info to fetch particle "
                    "semantics for each voxel.")
            assert sparse_semantics_event is not None, (
                    "Need to provide a semantics tensor to clean up output")
            sem_voxels, sem_features, _ = (
                    self.sparse_parser.process(sparse_semantics_event))
            np_voxels, np_features = (
                    clean_sparse_data(np_voxels, np_features, sem_voxels))

            # Match the semantic column to the reference tensor
            np_features[:, -1] = sem_features[:, -1]

            # Set all cluster labels to -1 if semantic class is LE or ghost
            shape_mask = sem_features[:, -1] > DELTA_SHP
            np_features[shape_mask, 1:-1] = -1 

            # If a value tree is provided, override value colum
            if sparse_value_event:
                _, val_features, _  = (
                        self.sparse_parser.process(sparse_value_event))
                np_features[:, 0] = val_features[:, -1]

        return np_voxels, np_features, Meta.from_larcv(meta)


class Cluster3DChargeRescaledParser(Cluster3DParser):
    """Identical to :class:`Cluster3DParser`, but computes rescaled charges
    on the fly. 
    """
    name = 'parse_cluster3d_rescale_charge'
    aliases = ['parse_cluster3d_charge_rescaled']

    def __init__(self, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        collection_only : bool, default False
            If True, only uses the collection plane charge
        collection_id : int, default 2
            Index of the collection plane
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Initialize the sparse parser which computes the rescaled charge
        from .sparse import Sparse3DChargeRescaledParser
        self.sparse_parser = Sparse3DChargeRescaledParser(
                collection_only, collection_id)

    def process(self, sparse_value_event_list, **kwargs):
        """Parse a list of 3D clusters into a single tensor.

        Parameters
        ----------
        sparse_value_event_list : List[larcv.EventSparseTensor3D]
            (6) List of sparse tensors used to compute the rescaled charge
            - Charge value of each of the contributing planes (3)
            - Index of the plane hit contributing to the space point (3)
        **kwargs : dict, optional
            Extra data products to pass to the parent Cluster3DParser

        Returns
        -------
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, 2/14) array of features, minimally [voxel value, cluster ID].
            If `add_particle_info` is `True`, the additonal columns are
            [group ID, interaction ID, neutrino ID, particle type,
            shower primary bool, primary group bool, vertex x, vertex y,
            vertex z, momentum, semantic type, particle ID]
        meta : Meta
            Metadata of the parsed image
        """
        # Process the input using the main parser
        np_voxels, np_features, meta = super().process(**kwargs)

        # Modify the value column using the charge rescaled on the fly
        _, val_features, _  = self.sparse_parser.process(
                sparse_value_event_list)
        np_features[:, 0] = val_features[:, -1]
        return np_voxels, np_features, meta


class Cluster3DMultiModuleParser(Cluster3DParser):
    """Identical to :class:`Cluster3DParser`, but fetches charge information
    from multiple modules independantly.
    """
    name = 'parse_cluster3d_multi_module'
    aliases = ['parse_cluster3d_2cryos']

    def process(self, sparse_value_event_list, **kwargs):
        """Parse a list of 3D clusters into a single tensor.

        Parameters
        ----------
        sparse_value_event_list : List[larcv.EventSparseTensor3D]
            (N_m) List of sparse value tensors, one per module
        **kwargs : dict, optional
            Extra data products to pass to the parent Cluster3DParser

        Returns
        -------
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, 2/14) array of features, minimally [voxel value, cluster ID].
            If `add_particle_info` is `True`, the additonal columns are
            [group ID, interaction ID, neutrino ID, particle type,
            shower primary bool, primary group bool, vertex x, vertex y,
            vertex z, momentum, semantic type, particle ID]
        meta : Meta
            Metadata of the parsed image
        """
        # Process the input using the main parser
        np_voxels, np_features, meta = super().process(**kwargs)

        # Fetch the charge information
        charges = np.zeros((len(np_voxels), 1), dtype=np.float32)
        for sparse_value_event in sparse_value_event_list:
            _, charges_i, _ = self.sparse_parser.process(sparse_value_event)
            charges[charges == 0.] = charges_i[charges == 0.]

        np_features[:, 0] = charges.flatten()

        return np_voxels, np_features, meta
