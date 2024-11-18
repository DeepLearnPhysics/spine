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

from spine.data import Meta, Particle, Neutrino, ObjectList

from spine.utils.globals import TRACK_SHP, PDG_TO_PID, PID_MASSES
from spine.utils.particles import process_particles
from spine.utils.ppn import get_ppn_labels, get_vertex_labels, image_coordinates
from spine.utils.conditional import larcv

from .base import ParserBase

__all__ = ['ParticleParser', 'NeutrinoParser', 'ParticlePointParser',
           'ParticleCoordinateParser', 'VertexPointParser', 'ParticleGraphParser',
           'SingleParticlePIDParser', 'SingleParticleEnergyParser']


class ParticleParser(ParserBase):
    """Class which loads larcv.Particle objects to local Particle ones.

    .. code-block. yaml

        schema:
          particles:
            parser: particle
            particle_event: particle_pcluster
            cluster_event: cluster3d_pcluster
            asis: False
            pixel_coordinates: True
            post_process: True
    """

    # Name of the parser (as specified in the configuration)
    name = 'particle'

    def __init__(self, pixel_coordinates=True, post_process=True,
                 skip_empty=False, asis=False, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        pixel_coordinates : bool, default True
            If set to `True`, the parser rescales the truth positions
            (start, end, etc.) to voxel coordinates
        post_process : bool, default True
            Processes particles to add/correct missing attributes
        skip_empty : bool, default False
            Do not read the truth information corresponding to empty particles.
            This saves considerable read time when there are a lot of irrelevant
            particle stored in the LArCV file. It puts an empty `Particle`
            object in place of empty particles, to preserve list size and typing.
        asis : bool, default False
            Load the objects as larcv objects, do not build local data class
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the revelant attributes
        self.pixel_coordinates = pixel_coordinates
        self.post_process = post_process
        self.skip_empty = skip_empty
        self.asis = asis

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(self, particle_event, sparse_event=None, cluster_event=None,
                particle_mpv_event=None, neutrino_event=None):
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
        particle_mpv_event : larcv.EventParticle, optional
            Particle event which contains the list of true MPV particles
        neutrino_event : larcv.EventNeutrino, optional
            Neutrino event which contains the list of true neutrinos

        Returns
        -------
        List[Particle]
            List of true particle objects
        """
        # If asis is true, return larcv objects
        particle_list = list(particle_event.as_vector())
        if self.asis:
            assert not self.pixel_coordinates, (
                    "If `asis` is True, `pixel_coordinates` must be False.")
            assert not self.post_process, (
                    "If `asis` is True, `post_process` must be False.")
            assert not self.skip_empty, (
                    "If `asis` is True`, `skip_empty` must be False.")

            return ObjectList(particle_list, larcv.Particle())

        # Convert to a list of particle objects
        particles = []
        for p in particle_list:
            if (not self.skip_empty or
                p.num_voxels() > 0 or
                p.id() == p.group_id()):
                particles.append(Particle.from_larcv(p))
            else:
                particles.append(Particle())

        # If requested, post-process the particle list
        if self.post_process:
            process_particles(particles, particle_event,
                              particle_mpv_event, neutrino_event)

        # If requested, convert the point positions to pixel coordinates
        if self.pixel_coordinates:
            # Fetch the metadata
            assert (sparse_event is not None) ^ (cluster_event is not None), (
                    "Must provide either `sparse_event` or `cluster_event` to "
                    "get the metadata and convert positions to voxel units.")
            ref_event = (
                    sparse_event if sparse_event is not None else cluster_event)
            meta = Meta.from_larcv(ref_event.meta())

            # Convert all the relevant attributes
            for p in particles:
                if p.id > -1:
                    p.to_px(meta)

        return ObjectList(particles, Particle())


class NeutrinoParser(ParserBase):
    """Class which loads larcv.Neutrino objects to local Neutrino ones.

    .. code-block. yaml

        schema:
          neutrinos:
            parser: neutrino
            neutrino_event: neutrino_mpv
            cluster_event: cluster3d_pcluster
            pixel_coordinates: True
            asis: False
    """

    # Name of the parser (as specified in the configuration)
    name = 'neutrino'

    def __init__(self, pixel_coordinates=True, asis=False, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        pixel_coordinates : bool, default True
            If set to `True`, the parser rescales the truth positions
            (start, end, etc.) to voxel coordinates
        asis : bool, default False
            Load the objects as larcv objects, do not build local data class
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the revelant attributes
        self.pixel_coordinates = pixel_coordinates
        self.asis = asis

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

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
        # If asis is true, return larcv objects
        neutrino_list = list(neutrino_event.as_vector())
        if self.asis:
            assert not self.pixel_coordinates, (
                    "If `asis` is True, `pixel_coordinates` must be False.")

            return ObjectList(neutrino_list, larcv.Neutrino())

        # Convert to a list of neutrino objects
        neutrinos = [Neutrino.from_larcv(n) for n in neutrino_list]

        # If requested, convert the point positions to pixel coordinates
        if self.pixel_coordinates:
            # Fetch the metadata
            assert (sparse_event is not None) ^ (cluster_event is not None), (
                    "Must provide either `sparse_event` or `cluster_event` to "
                    "get the metadata and convert positions to voxel units.")
            ref_event = (
                    sparse_event if sparse_event is not None else cluster_event)
            meta = Meta.from_larcv(ref_event.meta())

            # Convert all the relevant attributes
            for n in neutrinos:
                n.to_px(meta)

        return ObjectList(neutrinos, Neutrino())


class ParticlePointParser(ParserBase):
    """Class that retrieves the points of interests.

    It provides the coordinates of the end points, types and particle index.

    .. code-block. yaml

        schema:
          points:
            parser: particle_points
            particle_event: particle_pcluster
            sparse_event: sparse3d_pcluster
            include_point_tagging: True
    """

    # Name of the parser (as specified in the configuration)
    name = 'particle_points'

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

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

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
        ref_event = sparse_event if sparse_event is not None else cluster_event
        meta = ref_event.meta()

        # Get the point labels
        particle_v = particle_event.as_vector()
        point_labels = get_ppn_labels(
                particle_v, meta, self.ftype,
                include_point_tagging=self.include_point_tagging)

        return point_labels[:, :3], point_labels[:, 3:], Meta.from_larcv(meta)


class ParticleCoordinateParser(ParserBase):
    """Class that retrieves that end points of particles.

    It provides the coordinates of the end points, time and shape.

    .. code-block. yaml

        schema:
          coords:
            parser: particle_coordinates
            particle_event: particle_pcluster
            sparse_event: sparse3d_pcluster
    """

    # Name of the parser (as specified in the configuration)
    name = 'particle_coords'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

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
            (N, 6) array of [x_s, y_s, z_s, x_e, y_e, z_e] start and end
            point coordinates
        np_features : np.ndarray
            (N, 2) array of [first_step_t, shape_id]
        meta : Meta
            Metadata of the parsed image
        """
        # Fetch the metadata
        assert (sparse_event is not None) ^ (cluster_event is not None), (
                "Must provide either `sparse_event` or `cluster_event` to "
                "get the metadata and convert positions to voxel units.")
        ref_event = sparse_event if sparse_event is not None else cluster_event
        meta = ref_event.meta()

        # Scale particle coordinates to image size
        particle_v = particle_event.as_vector()

        # Make features
        features = np.empty((len(particle_v), 8), dtype=self.ftype)
        for i, p in enumerate(particle_v):
            start_point = last_point = image_coordinates(meta, p.first_step())
            if p.shape() == TRACK_SHP: # End point only meaningful for tracks
                last_point = image_coordinates(meta, p.last_step())
            extra = [p.t(), p.shape()]
            features[i] = np.concatenate((start_point, last_point, extra))

        return features[:, :6], features[:, 6:], Meta.from_larcv(meta)


class VertexPointParser(ParserBase):
    """Class that retrieves the vertices.

    It provides the coordinates of points where multiple particles originate:
    - If the `neutrino_event` is provided, it simply uses the coordinates of
      the neutrino interaction points.
    - If the `particle_event` is provided instead, it looks for ancestor point
      positions shared by at least two particles.

    .. code-block. yaml

        schema:
          vertices:
            parser: vertex_points
            particle_event: particle_pcluster
            #neutrino_event: neutrino_mpv
            sparse_event: sparse3d_pcluster
            include_point_tagging: True
    """

    # Name of the parser (as specified in the configuration)
    name = 'vertex_points'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(self, particle_event=None, neutrino_event=None,
                sparse_event=None, cluster_event=None):
        """Fetch the list of label vertex points.

        Parameters
        ----------
        particle_event : larcv.EventParticle
            Particle event which contains the list of true particles
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
        np_voxels : np.ndarray
            (N, 3) array of [x, y, z] coordinates
        np_features : np.ndarray
            (N, 1) array of [vertex ID]
        meta : Meta
            Metadata of the parsed image
        """
        # Check that only one source of vertex is provided
        assert (particle_event is not None) ^ (neutrino_event is not None), (
                "Must provide either `particle_event` or `sparse_event` to "
                "get the vertex points, not both.")

        # Fetch the metadata
        assert (sparse_event is not None) ^ (cluster_event is not None), (
                "Must provide either `sparse_event` or `cluster_event` to "
                "get the metadata and convert positions to voxel units.")
        ref_event = sparse_event if sparse_event is not None else cluster_event
        meta = ref_event.meta()

        # Get the vertex labels
        particle_v = particle_event.as_vector() if particle_event else None
        neutrino_v = neutrino_event.as_vector() if neutrino_event else None
        point_labels = get_vertex_labels(
                particle_v, neutrino_v, meta, self.ftype)

        return point_labels[:, :3], point_labels[:, 3:], Meta.from_larcv(meta)


class ParticleGraphParser(ParserBase):
    """Class that uses larcv.EventParticle to construct edges
    between particles (i.e. clusters).

    .. code-block. yaml

        schema:
          graph:
            parser: particle_graph
            particle_event: particle_pcluster
            cluster_event: cluster3d_pcluster

    """

    # Name of the parser (as specified in the configuration)
    name = 'particle_graph'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

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
            (2, E) Array of directed edges for each [parent, child] connection
        int
            Number of particles in the input
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
            if not edges:
                return np.empty((2, 0), dtype=np.int64), num_particles

            edges = np.vstack(edges).astype(np.int64)

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
            if not edges:
                return np.empty((2, 0), dtype=np.int64), num_particles

            edges = np.vstack(edges).astype(np.int64)

            # Remove zero pixel nodes
            for zn in zero_nodes:
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

        return edges.T, num_particles


class SingleParticlePIDParser(ParserBase):
    """Get the first true particle's species.

    .. code-block. yaml

        schema:
          image_pid:
            parser: single_particle_pid
            particle_event: particle_pcluster
    """

    # Name of the parser (as specified in the configuration)
    name = 'single_particle_pid'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

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
        pid = -1
        for p in particle_event.as_vector():
            if p.track_id() == 1:
                if int(p.pdg_code()) in PDG_TO_PID.keys():
                    pid = PDG_TO_PID[int(p.pdg_code())]

                break

        return pid


class SingleParticleEnergyParser(ParserBase):
    """Get the first true particle's kinetic energy.

    .. code-block. yaml

        schema:
          image_energy:
            parser: single_particle_energy
            particle_event: particle_pcluster
    """

    # Name of the parser (as specified in the configuration)
    name = 'single_particle_energy'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

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
