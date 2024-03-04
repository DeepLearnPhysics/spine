"""Module that contains all parsers related to LArCV particle data.

Contains the following parsers:
- :class:`ParticleParser`
- :class:`NeutrinoParser`
- :class:`ParticlePointParser`
- :class:`ParticleCoordinateParser`
- :class:`ParticleGraphParser`
- :class:`ParticlePIDParser`
- :class:`ParticleEnergyParser`
"""

import numpy as np
from larcv import larcv

from mlreco.utils.globals import TRACK_SHP, PDG_TO_PID, PID_MASSES
from mlreco.utils.data_structures import Meta, Particle, Neutrino
from mlreco.utils.unwrap import Unwrapper
from mlreco.utils.ppn import get_ppn_labels

from .parser import Parser

__all__ = ['ParticleParser', 'NeutrinoParser', 'ParticlePointParser',
           'ParticleCoordinateParser', 'ParticleGraphParser',
           'SingleParticlePIDParser', 'SingleParticleEnergyParser']


class ParticleParser(Parser):
    """Class which loads larcv.Particle objects to local Particle ones.

    .. code-block. yaml

        schema:
          particles:
            parser: parse_particles
            particle_event: particle_pcluster
            cluster_event: cluster3d_pcluster
            voxel_coordinates: True
    """
    name = 'parse_particles'
    result = Unwrapper.Rule(method='list', default=Particle())

    def __init__(self, voxel_coordinates=True, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        voxel_coordinates : bool, default True
            If set to `True`, the parser rescales the truth positions
            (start, end, etc.) to voxel coordinates
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the revelant attributes
        self.voxel_coordinates = voxel_coordinates

    def process(self, particle_event, sparse_event=None, cluster_event=None):
        """Fetch the list of true particle objects.

        Parameters
        ----------
        particle_event : larcv.EventParticle
            Particle event which contains the list of true particles
        sparse_event : larcv.EventSparseTensor3D, optional
            Tensor which contains the metadata needed to convert the
            positions in voxel coordinates
        cluster_event : larcv.EventClusterVoxel3D, optional
            Cluster which contains the metadata needed to convert the
            positions in voxel coordinates

        Returns
        -------
        List[Particle]
            List of true particle objects
        """
        # Convert to a list of LArCV particle objects
        particles = [larcv.Particle(p) for p in particle_event.as_vector()]
        particles = [Particle.from_larcv(p) for p in particles]
        if self.voxel_coordinates:
            # Fetch the metadata
            assert (sparse_event is not None) ^ (cluster_event is not None), (
                    "Must provide either `sparse_event` or `cluster_event` to "
                    "get the metadata and convert positions to voxel units.")
            ref_event = sparse_event if sparse_event else cluster_event
            meta = Meta.from_larcv(ref_event.meta())

            # Convert all the relevant attributes
            for p in particles:
                p.to_pixel(meta)

        return particles


class NeutrinoParser(Parser):
    """Class which loads larcv.Neutrino objects to local Neutrino ones.

    .. code-block. yaml

        schema:
          neutrinos:
            parser: parse_neutrinos
            neutrino_event: neutrino_mpv
            cluster_event: cluster3d_pcluster
            voxel_coordinates: True
    """
    name = 'parse_neutrinos'
    result = Unwrapper.Rule(method='list', default=Neutrino)

    def __init__(self, voxel_coordinates=True, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        voxel_coordinates : bool, default True
            If set to `True`, the parser rescales the truth positions
            (start, end, etc.) to voxel coordinates
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the revelant attributes
        self.voxel_coordinates = voxel_coordinates
        self.pos_attrs = ['position']

    def process(self, neutrino_event, sparse_event=None, cluster_event=None):
        """Fetch the list of true neutrino objects.

        Parameters
        ----------
        neutrino_event : larcv.EventNeutrino
            Neutrino event which contains the list of true neutrinos
        sparse_event : larcv.EventSparseTensor3D, optional
            Tensor which contains the metadata needed to convert the
            positions in voxel coordinates
        cluster_event : larcv.EventClusterVoxel3D, optional
            Cluster which contains the metadata needed to convert the
            positions in voxel coordinates

        Returns
        -------
        List[Neutrino]
            List of true neutrino objects
        """
        # Convert to a list of LArCV neutrino objects
        neutrinos = [larcv.Neutrino(n) for n in neutrino_event.as_vector()]
        neutrinos = [Neutrino.from_larcv(n) for n in neutrinos]
        if self.voxel_coordinates:
            # Fetch the metadata
            assert (sparse_event is not None) ^ (cluster_event is not None), (
                    "Must provide either `sparse_event` or `cluster_event` to "
                    "get the metadata and convert positions to voxel units.")
            ref_event = sparse_event if sparse_event else cluster_event
            meta = Meta.from_larcv(ref_event.meta())

            # Convert all the relevant attributes
            for n in neutrinos:
                n.to_pixel(meta)

        return neutrinos


class ParticlePointParser(Parser):
    """Class that retrieves the points of interests.

    It provides the coordinates of the end points, types and particle index.

    .. code-block. yaml

        schema:
          points:
            parser: parse_particle_points
            particle_event: particle_pcluster
            sparse_event: sparse3d_pcluster
            include_point_tagging: True
    """
    name = 'parse_particle_points'
    result = Unwrapper.Rule(method='tensor', translate=True)

    def __init__(self, include_point_tagging=True, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        include_point_tagging : bool, default True
            If `True`, includes start vs end point tagging
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the revelant attributes
        self.include_point_tagging = include_point_tagging

        # Based on the parameters, define a default output
        shape = (0, 6 + include_point_tagging)
        self.result.default = np.empty(shape, dtype=np.float32)

    def process(self, particle_event, sparse_event=None, cluster_event=None):
        """Fetch the list of label points of interest.

        Parameters
        ----------
        particle_event : larcv.EventParticle
            Particle event which contains the list of true particles
        sparse_event : larcv.EventSparseTensor3D, optional
            Tensor which contains the metadata needed to convert the
            positions in voxel coordinates
        cluster_event : larcv.EventClusterVoxel3D, optional
            Cluster which contains the metadata needed to convert the
            positions in voxel coordinates
            
        Returns
        -------
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, 2/3) array of [point type, particle index(, end point tagging)]
        meta : Meta
            Metadata of the parsed image
        """
        # Fetch the metadata
        assert (sparse_event is not None) ^ (cluster_event is not None), (
                "Must provide either `sparse_event` or `cluster_event` to "
                "get the metadata and convert positions to voxel units.")
        ref_event = sparse_event if sparse_event else cluster_event
        meta = ref_event.meta()

        # Get the point labels
        particles_v = particle_event.as_vector()
        point_labels = get_ppn_labels(
                particles_v, meta,
                include_point_tagging=self.include_point_tagging)

        return point_labels[:, :3], point_labels[:, 3:], Meta.from_larcv(meta)


class ParticleCoordinateParser(Parser):
    """Class that retrieves that end points of particles.

    It provides the coordinates of the end points, time and shape.

    .. code-block. yaml

        schema:
          coords:
            parser: parse_particle_coordinates
            particle_event: particle_pcluster
            sparse_event: sparse3d_pcluster
    """
    name = 'parse_particle_coords'
    result = Unwrapper.Rule(method='tensor', translate=True,
                            default=np.empty((0, 12), dtype=np.float32))

    def process(self, particle_event, sparse_event=None, cluster_event=None):
        """Fetch the start/end point and time of each true particle.

        Parameters
        ----------
        particle_event : larcv.EventParticle
            Particle event which contains the list of true particles
        sparse_event : larcv.EventSparseTensor3D, optional
            Tensor which contains the metadata needed to convert the
            positions in voxel coordinates
        cluster_event : larcv.EventClusterVoxel3D, optional
            Cluster which contains the metadata needed to convert the
            positions in voxel coordinates
            
        Returns
        -------
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] start point coordinates
        np_features : np.ndarray
            (N, 8) array of [first_step_x, first_step_y, first_step_z,
            last_step_x, last_step_y, last_step_z, first_step_t, shape_id]
        meta : Meta
            Metadata of the parsed image
        """
        # Fetch the metadata
        assert (sparse_event is not None) ^ (cluster_event is not None), (
                "Must provide either `sparse_event` or `cluster_event` to "
                "get the metadata and convert positions to voxel units.")
        ref_event = sparse_event if sparse_event else cluster_event
        meta = ref_event.meta()

        # Scale particle coordinates to image size
        particles = parse_particles(particle_event, ref_event)

        # Make features
        features = []
        for i, p in enumerate(particles):
            fs = p.first_step()
            start_point = last_point = [fs.x(), fs.y(), fs.z()]
            if p.shape() == TRACK_SHP: # End point only meaningful for tracks
                ls = p.last_step()
                last_point = [ls.x(), ls.y(), ls.z()]
            extra = [fs.t(), p.shape()]
            features.append(np.concatenate((start_point, last_point, extra)))

        features = np.vstack(features)

        # TODO: Should this not be just features? Collation of it will not work
        # if the input is broken down into multiple modules.
        return features[:, :3], features[:, 3:], Meta.from_larcv(meta)


class ParticleGraphParser(Parser):
    """Class that uses larcv.EventParticle to construct edges
    between particles (i.e. clusters).

    .. code-block. yaml

        schema:
          graph:
            parser: parse_particle_graph
            particle_event: particle_pcluster
            cluster_event: cluster3d_pcluster

    """
    name = 'parse_particle_graph'
    result = Unwrapper.Rule(method='tensor',
                            default=np.empty((0, 3), dtype=np.float32))

    def process(self, particle_event, cluster_event=None):
        """Fetch the parentage connections from the true particle list.

        Configuration
        -------------
        particle_event : larcv.EventParticle
            Particle event which contains the list of true particles
        cluster_event : larcv.EventClusterVoxel3D, optional
            Cluster used to check if particles have 0 pixel in the image. If
            so, the edges to those clusters are removed and the broken
            parantage is subsequently patched.

        Returns
        -------
        np.ndarray
            (E, 2) Array of directed edges for each [parent, child] connection
        """
        particles_v   = particle_event.as_vector()
        num_particles = particles_v.size()
        edges         = []
        if cluster_event is None:
            # Fill edges (directed [parent, child] pair)
            edges = []
            for cluster_id in range(num_particles):
                p = particles_v[cluster_id]
                if p.parent_id() != p.id():
                    edges.append([int(p.parent_id()), cluster_id])
                if p.parent_id() == p.id() and p.group_id() != p.id():
                    edges.append([int(p.group_id()), cluster_id])

            # Convert the list of edges to a numpy array
            if not len(edges):
                return np.empty((0, 2), dtype=np.int32)

            edges = np.vstack(edges).astype(np.int32)

        else:
            # Check that the cluster and particle objects are consistent
            num_clusters = cluster_event.size()
            assert (num_particles == num_clusters or
                    num_particles == num_clusters - 1), (
                    f"The number of particles ({num_particles}) must be "
                    f"aligned with the number of clusters ({num_clusters}). "
                    f"There can me one more catch-all cluster at the end.")

            # Fill edges (directed [parent, child] pair)
            zero_nodes, zero_nodes_pid = [], []
            for cluster_id in range(num_particles):
                cluster = cluster_event.as_vector()[cluster_id]
                num_points = cluster.as_vector().size()
                p = particles_v[cluster_id]
                if p.id() != p.group_id():
                    continue
                if p.parent_id() != p.group_id():
                    edges.append([int(p.parent_id()), p.group_id()])
                if num_points == 0:
                    zero_nodes.append(p.group_id())
                    zero_nodes_pid.append(cluster_id)

            # Convert the list of edges to a numpy array
            if not len(edges):
                return np.empty((0, 2), dtype=np.int32)

            edges = np.vstack(edges).astype(np.int32)

            # Remove zero pixel nodes
            for i, zn in enumerate(zero_nodes):
                children = np.where(edges[:, 0] == zn)[0]
                if len(children) == 0:
                    edges = edges[edges[:, 0] != zn]
                    edges = edges[edges[:, 1] != zn]
                    continue
                parent = np.where(edges[:, 1] == zn)[0]
                assert len(parent) <= 1

                # If zero node has a parent, then assign children to that parent
                if len(parent) == 1:
                    parent_id = edges[parent][0][0]
                    edges[:, 0][children] = parent_id
                else:
                    edges = edges[edges[:, 0] != zn]
                edges = edges[edges[:, 1] != zn]

        return edges


class SingleParticlePIDParser(Parser):
    """Get the first true particle's species.

    .. code-block. yaml

        schema:
          pdg_list:
            parser: parse_single_particle_pdg
            particle_event: particle_pcluster
    """
    name = 'parse_single_particle_pdg'
    aliases = ['parse_particle_singlep_pdg']
    result = Unwrapper.Rule(method='scalar', default=np.int32)

    def process(self, particle_event):
        """Fetch the species of the first particle.

        Configuration
        -------------
        particle_event : larcv.EventParticle
            Particle event which contains the list of true particles

        Returns
        -------
        int
            Species of the first particle
        """
        pdg = -1
        for p in particle_event.as_vector():
            if p.track_id() == 1:
                if int(p.pdg_code()) in PDG_TO_PID.keys():
                    pdg = PDG_TO_PID[int(p.pdg_code())]

                break

        return pdg


class SingleParticleEnergyParser(Parser):
    """Get the first true particle's kinetic energy.

    .. code-block. yaml

        schema:
          energy_list:
            parser: parse_single_particle_energy
            particle_event: particle_pcluster
    """
    name = 'parse_single_particle_energy'
    aliases = ['parse_particle_singlep_enit']
    result = Unwrapper.Rule(method='scalar', default=np.float32)

    def process(self, particle_event):
        """Fetch the kinetic energy of the first particle.

        Configuration
        -------------
        particle_event : larcv.EventParticle
            Particle event which contains the list of true particles

        Returns
        -------
        float
            Kinetic energy of the first particle
        """
        ke = -1.
        for p in particle_event.as_vector():
            if p.track_id() == 1:
                if int(p.pdg_code()) in PDG_TO_PID.keys():
                    einit = p.energy_init()
                    pid = PDG_TO_PID[int(p.pdg_code())]
                    mass = PID_MASSES[pid]
                    ke = einit - mass

                break

        return ke
