"""Module that supports barycenter-based flash matching."""

import numpy as np

from spine.utils.numba_local import cdist


class BarycenterFlashMatcher:
    """Matches interactions and flashes by matching the charge barycenter of
    TPC interactions with the light barycenter of optical flashes.
    """

    # List of valid matching methods
    _match_methods = ('threshold', 'best')

    def __init__(self, match_method='threshold', dimensions=[1, 2],
                 charge_weighted=False, time_window=None, first_flash_only=False,
                 min_inter_size=None, min_flash_pe=None, match_distance=None):
        """Initalize the barycenter flash matcher.

        Parameters
        ----------
        match_method: str, default 'distance'
            Matching method (one of 'threshold' or 'best')
            - 'threshold': If the two barycenters are within some distance, match
            - 'best': For each flash, pick the best matched interaction
        dimensions: list, default [1, 2]
            Dimensions involved in the distance computation
        charge_weighted : bool, default False
            Use interaction pixel charge information to weight the centroid
        time_window : List, optional
            List of [min, max] values of optical flash times to consider
        first_flash_only : bool, default False
            Only try to match the first flash in the time window
        min_inter_size : int, optional
            Minimum number of voxel in an interaction to consider it
        min_flash_pe : float, optional
            Minimum number of total PE in a flash to consider it
        match_distance : float, optional
            If a threshold is used, specifies the acceptable distance
        """
        # Check validity of key parameters
        if match_method not in self._match_methods:
            raise ValueError(
                     "Barycenter flash matching method not recognized: "
                    f"{match_method}. Must be one of {self._match_methods}.")

        if match_method == 'threshold':
            assert match_distance is not None, (
                    "When using the `threshold` method, must specify the "
                    "`match_distance` parameter.")

        # Store the flash matching parameters
        self.match_method     = match_method
        self.dims             = dimensions
        self.charge_weighted  = charge_weighted
        self.time_window      = time_window
        self.first_flash_only = first_flash_only
        self.min_inter_size   = min_inter_size
        self.min_flash_pe     = min_flash_pe
        self.match_distance   = match_distance


    def get_matches(self, interactions, flashes):
        """Makes [interaction, flash] pairs that have compatible barycenters.

        Parameters
        ----------
        interactions : List[Union[RecoInteraction, TruthInteraction]]
            List of interactions
        flashes : List[Flash]
            List of optical flashes

        Returns
        -------
        List[[Interaction, larcv.Flash, float]]
            List of [interaction, flash, distance] triplets
        """
        # Restrict the flashes to those that fit the selection criteria.
        # Skip if there are no valid flashes
        if self.time_window is not None:
            t1, t2 = self.time_window
            flahses = [f for f in flashes if (f.time > t1 and f.time < t2)]

        if self.min_flash_pe is not None:
            flashes = [f for f in flashes if f.total_pe > self.min_flash_pe]

        if not len(flashes):
            return []

        # If requested, restrict the list of flashes to match to the first one
        if self.first_flash_only:
            flashes = [flashes[0]]

        # Restrict the interactions to those that fit the selection criterion.
        # Skip if there are no valid interactions
        if self.min_inter_size is not None:
            interactions = [
                    inter for inter in interactions
                    if inter.size > self.min_inter_size
            ]

        if not len(interactions):
            return []

        # Get the flash centroids
        op_centroids = np.empty((len(flashes), len(self.dims)))
        op_widths = np.empty((len(flashes), len(self.dims)))
        for i, f in enumerate(flashes):
            op_centroids[i] = f.center[self.dims]
            op_widhts[i] = f.width[self.dims]

        # Get interactions centroids
        int_centroids = np.empty((len(interactions), len(self.dims)))
        for i, inter in enumerate(interactions):
            if not self.charge_weighted:
                int_centroids[i] = np.mean(inter.points[:, self.dims], axis=0)

            else:
                int_centroids[i] = np.sum(
                        inter.depositions * inter.points[:, self.dims], axis=0)
                int_centroids[i] /= np.sum(inter.depositions)

        # Compute the flash to interaction distance matrix
        dist_mat = cdist(op_centroids, int_centroids)

        # Produce matches
        matches = []
        if self.match_method == 'best':
            # For each flash, select the best match, save attributes
            for i, f in enumerate(flashes):
                best_match = np.argmin(dist_mat[i])
                dist = dist_mat[i, best_match]
                if (self.match_distance is not None and
                    dist > self.match_distance):
                        continue
                match.append((interactions[best_match], f, dist))

        elif self.match_method == 'threshold':
            # Find all compatible pairs
            valid_pairs = np.vstack(np.where(dist_mat <= self.match_distance)).T
            matches = [
                    (interactions[j], flashes[i], dist_mat[i,j])
                    for i, j in valid_pairs
            ]

        return matches
