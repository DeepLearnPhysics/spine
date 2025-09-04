"""Module to evaluate diagnostic metrics on points."""

import numpy as np
from scipy.spatial import cKDTree
from spine.utils.globals import GROUP_COL, COORD_COLS

from spine.ana.base import AnaBase


__all__ = ['PointMetricsAna']


class PointMetricsAna(AnaBase):
    """This analysis script evaluates the purity and efficiency of spacepoints.
    Compare the truth points (sed) to the reconstructed points (cluster3d)."""

    # Name of the analysis script (as specified in the configuration)
    name = 'point_metrics'

    def __init__(self, time_window=None, run_mode='both',
                 truth_point_mode='points', **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        time_window : List[float]
            Time window within which to include particle (only works for `truth`)
        run_mode : str
            Mode to run the analysis in, either 'both', 'reco', or 'truth'
        truth_point_mode : str
            Mode to run the truth points in, either 'points' or 'clusters'
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__('particle', run_mode, truth_point_mode, **kwargs)

        # Store the time window
        self.time_window = time_window
        assert time_window is None or len(time_window) == 2, (
                "Time window must be specified as an array of two values.")
        assert time_window is None or run_mode == 'truth', (
                "Time of reconstructed particle is unknown.")

        # Make sure the metadata is provided (rasterization needed)
        self.update_keys({'meta': True})

        # Initialize the CSV writer(s) you want
        for prefix in self.prefixes:
            self.initialize_writer(prefix)

    def process(self, data):
        """Evaluate track completeness for tracks in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """

        # Fetch the pixel size in this image (assume cubic cells)
        pixel_size = data['meta'].size[0]

        # Loop over the types of particle data products
        for key in self.obj_keys:
            # Fetch the prefix ('reco' or 'truth')
            prefix = key.split('_')[0]

            # Loop over particle objects to collect all points
            for i,part in enumerate(data[key]):
                # If needed, check on the particle time
                if self.time_window is not None:
                    if part.t < self.time_window[0] or part.t > self.time_window[1]:
                        continue
                
                # Fetch the particle point coordinates
                points = self.get_points(part)
                sed_points = part.points_g4

                if len(points) == 0 or len(sed_points) == 0:
                    continue

                # Get the purity and efficiency
                # distance = 3*pixel_size, so we can match by the corner of the voxel
                purity, efficiency = self.reco_and_true_matching(points, sed_points, distance=3*pixel_size)

                # Initialize the particle dictionary
                comp_dict = {'particle_id': part.id}
                comp_dict['purity'] = purity
                comp_dict['efficiency'] = efficiency
                comp_dict['num_points'] = len(points)
                comp_dict['num_sed_points'] = len(sed_points)
                comp_dict['shape'] = part.shape

                # Add length for tracks
                if part.shape == 'track':
                    start = points[np.argmin(cdist([part.start_point], points))]
                    end = points[np.argmin(cdist([part.end_point], points))]
                    vec = end - start
                    length = np.linalg.norm(vec)
                    if length and length > 0:
                        vec /= length
                    comp_dict['length'] = length
                    comp_dict.update(
                        {'dir_x': vec[0], 'dir_y': vec[1], 'dir_z': vec[2]})

                # Append the dictionary to the CSV log
                self.append(prefix, **comp_dict)

    @staticmethod
    def reco_and_true_matching(reco_noghost,true,distance=3):
        """
        Calculates purtiy and efficiency by matching true voxel locations to reco voxel locations
        and vise-versa


        reco_noghost = xyz coords for nonghost
        true         = xyz for true parts
        distance     = threshold distance between voxels. Use 3 by default since this is a matching by the corner of the voxel.
                       Scale by pixel size if in cm

        eff                 = true tagged voxel count / true voxel count
        pur                 = reco tagged voxel count / reco voxels (noghost)
        reco_tagged         = reco voxels that were matched to a true voxel (true->reco)
        true_tagged         = true voxels that were matched to a reco voxel (reco->true)
        reco_reverse_tagged = reco voxels that were matched to a true voxel (reco->true)
        """
        small = 1e-5 #offset for float precision
        tree = cKDTree(true) #Get tree to perform query
        #Return closest distance to each reco point, and indices of that voxel in the truth array
        distances, indices = tree.query(reco_noghost,k=1)
        reco_indices = []
        for i,d in enumerate(distances): #distances from nearest true voxel
            d-=small
            if d<=distance: #Ignore elements that don't satisfy distance threshold
                reco_indices.append(i)
        if len(reco_indices) > 0:
            reco_tagged = reco_noghost[np.unique(reco_indices)]
            pur = len(reco_tagged)/len(reco_noghost)
        else:
            pur = 0

        #Do it again for efficiency
        tree = cKDTree(reco_noghost) #Get tree to perform query
        #Return closest distance to each reco voxel, and indices of that voxel in the truth array
        distances, indices = tree.query(true,k=1) #Distance from reco to true voxels
        true_indices = []
        reco_tagged_indices = [] #additional index set to store reco points that were matched to true
        for i,d in enumerate(distances):
            d-=small
            if d<=distance: #Ignore elements that don't satisfy the criteria
                true_indices.append(i)
                reco_tagged_indices.append(indices[i])
        if len(reco_indices) > 0:
            reco_reverse_tagged = reco_noghost[np.unique(reco_indices)] #Matched from truth voxel
            true_tagged = true[np.unique(true_indices)]
            eff = len(true_tagged)/len(true)
        else:
            eff = 0

        return pur,eff
            
    