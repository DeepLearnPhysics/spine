"""Shower reconstruction module."""

import numpy as np
from scipy.stats import pearsonr

from spine.utils.globals import SHOWR_SHP, TRACK_SHP, PROT_PID, PION_PID
from spine.utils.numba_local import cdist

from spine.post.base import PostBase

__all__ = ['ShowerConversionDistanceProcessor',
           'ShowerStartCorrectionProcessor']


class ShowerConversionDistanceProcessor(PostBase):
    """Evaluates the distance between shower start and the position
    of the vertex of the interaction they belong to.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'shower_conversion_distance'

    # List of recognized ways to compute the vertex-to-shower distance
    _modes = ('vertex_to_start', 'vertex_to_points', 'protons_to_points')

    def __init__(self, mode='vertex_to_points', include_secondary=False):
        """Store the conversion distance reconstruction parameters.

        Parameters
        ----------
        mode : str, default 'vertex'
            Method used to compute the conversion distance:
            - 'vertex_to_start': Distance between the vertex and the predicted
               shower start point.
            - 'vertex_to_points': Shortest distance between the vertex and any
              shower point.
            - 'protons_to_points': Distance between any proton/pion point and
              any shower point.
        include_secondary : bool, default False
            If `True`, computes the covnersion distance for secondary particles
        """
        # Initialize the parent class
        super().__init__('interaction', 'reco')

        # Check that the conversion distance mode is valid, store
        assert mode in self._modes, (
                f"Conversion distance computation mode not recognized: {mode}. "
                f"Should be one of {self._modes}")
        self.mode = mode

        # Store the vertex distance computation parameters
        self.include_secondary = include_secondary

    def process(self, data):
        """Compute the conversion distance of showers in each interaction.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for inter in data['reco_interactions']:
            # Loop over the particles that make up the interaction
            for part in inter.particles:
                # If the particle is a secondary and they are to be skipped, do
                if not self.include_secondary and not part.is_primary:
                    continue

                # If the particle PID is not a shower, skip
                if part.shape != SHOWR_SHP:
                    continue

                # Compute the vertex distance
                if self.mode == 'vertex_to_start':
                    dist = self.get_vertex_to_start(inter, part)
                elif self.mode == 'vertex_to_points':
                    dist = self.get_vertex_to_points(inter, part)
                elif self.mode == 'protons_to_start':
                    dist = self.get_protons_to_points(inter, part)

                part.vertex_distance = dist

    @staticmethod
    def get_vertex_to_start(inter, shower):
        """Helper function to compute the closest distance between the vertex
        and the predicted shower startpoint.

        Parameters
        ----------
        inter : RecoInteraction
            Reco interaction providing the vertex
        shower : RecoParticle
            Shower particle in the interaction

        Returns
        -------
        float
            Distance betweenthe vertex and the shower start point
        """
        # Compute distance from the vertex to the shower start point
        dist = np.linalg.norm(inter.vertex - shower.start_point)

        return dist

    @staticmethod
    def get_vertex_to_points(inter, shower):
        """Helper function to compute the closest distance between the vertex
        and all shower points.

        Parameters
        ----------
        inter : RecoInteraction
            Reco interaction providing the vertex
        shower : RecoParticle
            Shower particle in the interaction

        Returns
        -------
        float
            Shortest distance between the vertex and any shower point
        """
        # Compute distances from the vertex to all shower points, find minimum
        dists = cdist(inter.vertex.reshape(-1, 3), shower.points)

        return dists.min()

    @staticmethod
    def get_protons_to_points(inter, shower):
        """Helper function to compute the closest distance between any
        proton/pion point and any shower point.

        Parameters
        ----------
        inter : RecoInteraction
            Reco interaction providing the vertex
        shower : RecoParticle
            Shower particle in the interaction

        Returns
        -------
        float
            Shortest distance between the shower and proton/pion points
        """
        # Fetch the list of points associated with primary pions or protons
        proton_points = []
        for part in inter.particles:
            if part.pid in (PION_PID, PROT_PID) and part.is_primary:
                proton_points.append(part.points)

        # If there is no pion/proton point, return dummy value
        if len(proton_points) == 0:
            return -1

        # Compute conversion distance
        proton_points = np.vstack(proton_points)
        dists = cdist(proton_points, shower.points)

        return dists.min()


class ShowerStartCorrectionProcessor(PostBase):
    """Correct the start point of primary EM showers by finding the closest
    point to the vertex.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'shower_start_correction'

    def __init__(self):
        """Store the shower start corrector parameters."""
        # Initialize the parent class
        super().__init__('interaction', 'reco')

    def process(self, data):
        """Update the shower start point using the closest point to the vertex.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for inter in data['reco_interactions']:
            # Loop over the particles that make up the interaction
            for part in inter.particles:
                # If the particle PID is not a shower or not primary, skip
                if part.shape != SHOWR_SHP or not part.is_primary:
                    continue

                # Override the start point attribute of the particle
                part.start_point = self.correct_shower_start(inter, part)

    @staticmethod
    def correct_shower_start(inter, shower):
        """Function to correct the shower start point by finding the closest
        point to any primary track in the image.

        Parameters
        ----------
        inter : RecoInteraction
            Reco interaction to provide a vertex estimate
        shower : RecoParticle
            Primary EM shower to find the shower start point for

        Returns
        -------
        guess : np.ndarray
            (3, ) array of the corrected shower start point.
        """
        # Get the set of primary track points
        track_points = []
        for part in inter.particles:
            if part.shape == TRACK_SHP and part.is_primary:
                track_points.append(part.points)

        # If there are no track points in the interaction, keep existing
        if len(track_points) == 0:
            return shower.start_point

        # Update the shower start point by finding the consitituent point
        # closest to any of the track points
        track_points = np.vstack(track_points)
        dists = np.min(cdist(shower.points, track_points), axis=1)
        closest_idx = np.argmin(dists)
        start_point = shower.points[closest_idx]

        return start_point
