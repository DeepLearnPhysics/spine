"""Detector-geometry-based reconstruction module."""

import numpy as np

from spine.utils.globals import PID_LABELS
from spine.utils.geo import Geometry

from spine.post.base import PostBase

__all__ = ['ContainmentProcessor', 'FiducialProcessor']


class ContainmentProcessor(PostBase):
    """Check whether a fragment, particle or interaction is contained.

    This module checks whether the object comes within some distance of the
    boundaries of the detector and assign the `is_contained` attribute
    accordingly.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'containment'

    # Alternative allowed names of the post-processor
    aliases = ('check_containment',)

    def __init__(self, margin, cathode_margin=None, detector=None,
                 geometry_file=None, mode='module',
                 allow_multi_module=False, min_particle_sizes=0,
                 obj_type=('particle', 'interaction'),
                 truth_point_mode='points', run_mode='both'):
        """Initialize the containment conditions.

        If the `source` method is used, the cut will be based on the source of
        the point cloud, i.e. if a point cloud was produced by TPCs i and j, it
        must be contained within the volume bound by the set of TPCs i and j,
        and whichever volume is present between them.

        Parameters
        ----------
        margin : Union[float, List[float], np.array]
            Minimum distance from a detector wall to be considered contained:
            - If float: distance buffer is shared between all 6 walls
            - If [x,y,z]: distance is shared between pairs of falls facing
              each other and perpendicular to a shared axis
            - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is
              specified individually of each wall.
        cathode_margin : float, optional
            If specified, sets a different margin for the cathode boundaries
        detector : str, optional
            Detector to get the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        mode : str, default 'module'
            Containement criterion (one of 'global', 'module', 'tpc'):
            - If 'tpc', makes sure it is contained within a single tpc
            - If 'module', makes sure it is contained within a single module
            - If 'detector', makes sure it is contained within the
              outermost walls
            - If 'source', use the origin of voxels to determine which TPC(s)
              contributed to them, and define volumes accordingly
            - If 'meta', use the metadata range as a basis for containment.
              Note that this does not guarantee containment within the detector.
        allow_multi_module : bool, default False
            Whether to allow particles/interactions to span multiple modules
        min_particle_sizes : Union[int, dict], default 0
            When checking interaction containment, ignore particles below the
            size (in voxel count) specified by this parameter. If specified
            as a dictionary, it maps a specific particle type to its own cut.
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode)

        # Initialize the geometry, if needed
        if mode != 'meta':
            self.use_meta = False
            self.geo = Geometry(detector, geometry_file)
            self.geo.define_containment_volumes(margin, cathode_margin, mode)

        else:
            assert detector is None and geometry_file is None, (
                    "When using `meta` to check containment, must not "
                    "provide geometry information.")
            self.update_keys({'meta': True})
            self.use_meta = True
            self.margin = margin

        # Store parameters
        self.allow_multi_module = allow_multi_module

        # Store the particle size thresholds in a dictionary
        if np.isscalar(min_particle_sizes):
            min_particle_sizes = {'default': min_particle_sizes}

        self.min_particle_sizes = {}
        for pid in PID_LABELS.keys():
            if pid in min_particle_sizes:
                self.min_particle_sizes[pid] = min_particle_sizes[pid]
            elif 'default' in min_particle_sizes:
                self.min_particle_sizes[pid] = min_particle_sizes['default']
            else:
                self.min_particle_sizes[pid] = 0

    def process(self, data):
        """Check the containment of all objects in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Get the metadata information, if need be
        if self.use_meta:
            meta = data['meta']

        # Loop over particle objects
        for k in self.fragment_keys + self.particle_keys:
            for obj in data[k]:
                # Make sure the particle coordinates are expressed in cm
                self.check_units(obj)

                # Get point coordinates
                points = self.get_points(obj)
                if not len(points):
                    obj.is_contained = True
                    continue

                # Check particle containment
                if not self.use_meta:
                    obj.is_contained = self.geo.check_containment(
                            points, obj.sources, self.allow_multi_module)
                else:
                    obj.is_contained = (
                            (points > (meta.lower + self.margin)).all() and
                            (points < (meta.upper - self.margin)).all())

        # Loop over interaction objects
        for k in self.interaction_keys:
            for inter in data[k]:
                # Check that all the particles in the interaction are contained
                inter.is_contained = True
                for part in inter.particles:
                    if not part.is_contained:
                        # Do not account for particles below a certain size
                        if (part.pid > -1 and
                            part.size < self.min_particle_sizes[part.pid]):
                            continue

                        inter.is_contained = False
                        break


class FiducialProcessor(PostBase):
    """Check that an interaction vertex is in a user-defined fiducial volume.

    The fiducial volume is defined as a margin distances from each of the
    detector walls.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'fiducial'

    # Alternative allowed names of the post-processor
    aliases = ('check_fiducial',)

    def __init__(self, margin, cathode_margin=None, detector=None,
                 geometry_file=None, mode='module', run_mode='both',
                 truth_vertex_mode='vertex'):
        """Initialize the fiducial conditions.

        Parameters
        ----------
        margin : Union[float, List[float], np.array]
            Minimum distance from a detector wall to be considered contained:
            - If float: distance buffer is shared between all 6 walls
            - If [x,y,z]: distance is shared between pairs of falls facing
              each other and perpendicular to a shared axis
            - If [[x_low,x_up], [y_low,y_up], [z_low,z_up]]: distance is
              specified individually of each wall.
        cathode_margin : float, optional
            If specified, sets a different margin for the cathode boundaries
        detector : str, default 'icarus'
            Detector to get the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        mode : str, default 'module'
            Containement criterion (one of 'global', 'module', 'tpc'):
            - If 'tpc', makes sure it is contained within a single tpc
            - If 'module', makes sure it is contained within a single module
            - If 'detector', makes sure it is contained within the
              outermost walls
            - If 'meta', use the metadata range as a basis for containment.
              Note that this does not guarantee containment within the detector.
        truth_vertex_mode : str, default 'truth_vertex'
             Vertex attribute to use to check containment of true interactions
        """
        # Initialize the parent class
        super().__init__('interaction', run_mode)

        # Initialize the geometry
        if mode != 'meta':
            self.use_meta = False
            self.geo = Geometry(detector, geometry_file)
            self.geo.define_containment_volumes(margin, cathode_margin, mode)

        else:
            assert detector is None and geometry_file is None, (
                    "When using `meta` to check containment, must not "
                    "provide geometry information.")
            self.update_keys({'meta': True})
            self.use_meta = True
            self.margin = margin

        # Store the true vertex source
        assert truth_vertex_mode in ['vertex', 'reco_vertex'], (
                f"`truth_vertex_mode not recognized: `{truth_vertex_mode}`. "
                 "Must be one one of `vertex` or `reco_vertex`.")
        self.truth_vertex_mode = truth_vertex_mode

    def process(self, data):
        """Check the fiducial status of all interactions in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Get the metadata information, if need be
        if self.use_meta:
            meta = data['meta']

        # Loop over interaction objects
        for k in self.interaction_keys:
            for inter in data[k]:
                # Make sure the interaction coordinates are expressed in cm
                self.check_units(inter)

                # Get point coordinates
                if not inter.is_truth:
                    vertex = inter.vertex
                else:
                    vertex = getattr(inter, self.truth_vertex_mode)
                vertex = vertex.reshape(-1,3)

                # Check containment
                if not self.use_meta:
                    inter.is_fiducial = self.geo.check_containment(vertex)
                else:
                    inter.is_fiducial = (
                            (vertex > (meta.lower + self.margin)).all() and
                            (vertex < (meta.upper - self.margin)).all())
