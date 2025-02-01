"""Module to evaluate diagnostic metrics on tracks."""

import numpy as np
from scipy.spatial.distance import cdist

from spine.ana.base import AnaBase

from spine.utils.globals import TRACK_SHP
from spine.utils.numba_local import principal_components


__all__ = ['TrackCompletenessAna']


class TrackCompletenessAna(AnaBase):
    """This analysis script identifies gaps in tracks and measures the
    cumulative length of these gaps relative to the track length.

    This is a useful diagnostic tool to evaluate the space-point efficiency
    on tracks (good standard candal as track should have exactly no gap in
    a perfectly efficient detector).
    """

    # Name of the analysis script (as specified in the configuration)
    name = 'track_completeness'

    def __init__(self, time_window=None, run_mode='both',
                 truth_point_mode='points', **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        time_window : List[float]
            Time window within which to include particle (only works for `truth`)
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

            # Loop over particle objects
            for part in data[key]:
                # Check that the particle is a track
                if part.shape != TRACK_SHP:
                    continue

                # If needed, check on the particle time
                if self.time_window is not None:
                    if part.t < self.time_window[0] or part.t > self.time_window[1]:
                        continue

                # Initialize the particle dictionary
                comp_dict = {'particle_id': part.id}

                # Fetch the particle point coordinates
                points = self.get_points(part)

                # Find start/end points, collapse onto track cluster
                start = points[np.argmin(cdist([part.start_point], points))]
                end = points[np.argmin(cdist([part.end_point], points))]

                # Add the direction of the track
                vec = end - start
                length = np.linalg.norm(vec)
                if length:
                    vec /= length

                comp_dict['size'] = len(points)
                comp_dict['length'] = length
                comp_dict.update(
                        {'dir_x': vec[0], 'dir_y': vec[1], 'dir_z': vec[2]})

                # Chunk out the track along gaps, estimate gap length
                chunk_labels = self.cluster_track_chunks(
                        points, start, end, pixel_size)
                gaps = self.sequential_cluster_distances(
                        points, chunk_labels, start)

                # Substract minimum gap distance due to rasterization
                min_gap = pixel_size/np.max(np.abs(vec))
                gaps -= min_gap

                # Store gap information
                comp_dict['num_gaps'] = len(gaps)
                comp_dict['gap_length'] = np.sum(gaps)
                comp_dict['gap_frac'] = np.sum(gaps)/length

                # Append the dictionary to the CSV log
                self.append(prefix, **comp_dict)

    @staticmethod
    def cluster_track_chunks(points, start_point, end_point, pixel_size):
        """Find point where the track is broken, divide out the track
        into self-contained chunks which are Linf connect (Moore neighbors).

        Parameters
        ----------
        points : np.ndarray
            (N, 3) List of track cluster point coordinates
        start_point : np.ndarray
            (3) Start point of the track cluster
        end_point : np.ndarray
            (3) End point of the track cluster
        pixel_size : float
            Dimension of one pixel, used to identify what is big enough to
            constitute a break

        Returns
        -------
        np.ndarray
            (N) Track chunk labels
        """
        # Project and cluster on the projected axis
        direction = (end_point-start_point)/np.linalg.norm(end_point-start_point)
        projs = np.dot(points - start_point, direction)
        perm = np.argsort(projs)
        seps = projs[perm][1:] - projs[perm][:-1]
        breaks = np.where(seps > pixel_size*1.1)[0] + 1
        cluster_labels = np.empty(len(projs), dtype=int)
        for i, index in enumerate(np.split(np.arange(len(projs)), breaks)):
            cluster_labels[perm[index]] = i
            
        return cluster_labels

    @staticmethod
    def sequential_cluster_distances(points, labels, start_point):
        """Order clusters in order of distance from a starting point, compute
        the distances between successive clusters. 

        Parameters
        ----------
        points : np.ndarray
            (N, 3) List of track cluster point coordinates
        labels : np.ndarray
            (N) Track chunk labels
        start_point : np.ndarray
            (3) Start point of the track cluster
        """
        # If there's only one cluster, nothing to do here
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        
        # Order clusters
        start_dist = cdist([start_point], points).flatten()
        start_clust_dist = np.empty(len(unique_labels))
        for i, c in enumerate(unique_labels):
            start_clust_dist[i] = np.min(start_dist[labels == c])
        ordered_labels = unique_labels[np.argsort(start_clust_dist)]
        
        # Compute the intercluster distance and relative angle
        n_gaps = len(ordered_labels) - 1
        dists = np.empty(n_gaps, dtype=float)
        for i in range(n_gaps):
            points_i = points[labels == ordered_labels[i]]
            points_j = points[labels == ordered_labels[i + 1]]
            dists[i] = np.min(cdist(points_i, points_j))
            
        return dists
