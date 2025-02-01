"""Particle direction reconstruction module."""

from spine.utils.globals import TRACK_SHP, PID_LABELS
from spine.utils.gnn.cluster import cluster_direction

from spine.post.base import PostBase

__all__ = ['DirectionProcessor']


class DirectionProcessor(PostBase):
    """Reconstructs the direction of fragments and/or particles w.r.t. to
    their start (and end for tracks) points.

    This modules assign the `start_dir` and `end_dir` attributes.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'direction'

    # Alternative allowed names of the post-processor
    aliases = ('reconstruct_directions',)

    def __init__(self, neighborhood_radius=-1, optimize=True,
                 obj_type='particle', truth_point_mode='points',
                 run_mode='both'):
        """Store the particle direction recosntruction parameters.

        Parameters
        ----------
        neighborhood_radius : float, default 5
            Max distance between start voxel and other voxels
        optimize : bool, default True
            Optimizes the number of points involved in the estimate
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode)

        # Store the direction reconstruction parameters
        self.neighborhood_radius = neighborhood_radius
        self.optimize = optimize

    def process(self, data):
        """Reconstruct the directions of all particles in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle objects
        for k in self.obj_keys:
            for obj in data[k]:
                # Make sure the particle coordinates are expressed in cm
                self.check_units(obj)

                # Get point coordinates
                points = self.get_points(obj)
                if not len(points):
                    continue

                # Reconstruct directions from either end of the particle
                start_attr = 'reco_start_dir' if obj.is_truth else 'start_dir'
                setattr(obj, start_attr, cluster_direction(
                        points, obj.start_point, 
                        self.neighborhood_radius, self.optimize))
                if obj.shape == TRACK_SHP:
                    end_attr = 'reco_end_dir' if obj.is_truth else 'end_dir'
                    setattr(obj, end_attr, -cluster_direction(
                            points, obj.end_point,
                            self.neighborhood_radius, self.optimize))
