"""Shower reconstruction module."""

import numpy as np

from spine.data import ObjectList, RecoParticle
from spine.math.distance import cdist
from spine.post.base import PostBase
from spine.utils.globals import PION_PID, PROT_PID, SHOWR_SHP, TRACK_SHP
from spine.utils.gnn.cluster import cluster_dedx, cluster_direction

__all__ = [
    "ShowerConversionDistanceProcessor",
    "ShowerStartMergeProcessor",
    "ShowerStartCorrectionProcessor",
]


class ShowerConversionDistanceProcessor(PostBase):
    """Evaluates the distance between shower start and the position
    of the vertex of the interaction they belong to.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "shower_conversion_distance"

    # List of recognized ways to compute the vertex-to-shower distance
    _modes = ("vertex_to_start", "vertex_to_points", "protons_to_points")

    def __init__(self, mode="vertex_to_points", include_secondary=False):
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
        super().__init__("interaction", "reco")

        # Check that the conversion distance mode is valid, store
        assert mode in self._modes, (
            f"Conversion distance computation mode not recognized: {mode}. "
            f"Should be one of {self._modes}"
        )
        self.mode = mode

        # Store the vertex distance computation parameters
        self.include_secondary = include_secondary

        # If the method involves the vertex, must run the vertex PP
        if "vertex" in self.mode:
            self.update_upstream("vertex")

    def process(self, data):
        """Compute the conversion distance of showers in each interaction.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for inter in data["reco_interactions"]:
            # Loop over the particles that make up the interaction
            for part in inter.particles:
                # If the particle is a secondary and they are to be skipped, do
                if not self.include_secondary and not part.is_primary:
                    continue

                # If the particle PID is not a shower, skip
                if part.shape != SHOWR_SHP:
                    continue

                # Compute the vertex distance
                if self.mode == "vertex_to_start":
                    dist = self.get_vertex_to_start(inter, part)
                elif self.mode == "vertex_to_points":
                    dist = self.get_vertex_to_points(inter, part)
                elif self.mode == "protons_to_start":
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


class ShowerStartMergeProcessor(PostBase):
    """Merge shower start if it was classified as track points.

    This post-processor is used to patch upstream semantic segmentation mistakes
    where part of the shower trunk is assigned to track points and subsequantly
    classified as a separate particle.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "shower_start_merge"

    # Set of post-processors which must be run before this one is
    _upstream = ("direction",)

    def __init__(
        self,
        angle_threshold=0.175,
        distance_threshold=1.0,
        track_length_limit=50,
        track_dedx_limit=None,
        **kwargs,
    ):
        """Store the shower start merging parameters.

        Parameters
        ----------
        angle_threshold : float, default 0.175
            Track and shower angular agreement threshold to be merged (radians)
        distance_threshold : float, default 1.0
            Track and shower distance threshold to be merged (cm)
        track_length_limit : int, default 50
            Maximum track length allowed to mitigate merging legitimate tracks
        track_dedx_limit : float, optional
            Maximum track dE/dx allowed to mitigate proton merging (MeV/cm)
        """
        # Initialize the parent class
        super().__init__("interaction", "reco")

        # Store the shower start merging parameters
        self.angle_threshold = abs(np.cos(angle_threshold))
        self.distance_threshold = distance_threshold
        self.track_length_limit = track_length_limit
        self.track_dedx_limit = track_dedx_limit

    def process(self, data):
        """Merge split shower starts for all particles in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[RecoParticle]
            Updated set of particles (interactions modified in place)
        """
        # Loop over the reco interactions
        particles = ObjectList([], default=RecoParticle())
        offset = 0
        for inter in data["reco_interactions"]:
            # Fetch the leading shower in the interaction, skip if None
            shower = inter.leading_shower
            if shower is None:
                for part in inter.particles:
                    part.id = offset
                    offset += 1

                particles.extend(inter.particles)
                inter.particle_ids = np.array([part.id for part in inter.particles])
                continue

            # Merge tracks which fit the criteria into the leading shower
            merge_ids = []
            for part in inter.particles:
                # Only check mergeability of primary tracks
                if part.shape == TRACK_SHP and part.is_primary:
                    if self.check_merge(shower, part):
                        shower.merge(part)
                        merge_ids.append(part.id)

            # Update the particle list of the interaction
            new_particles = []
            for part in inter.particles:
                if part.id not in merge_ids:
                    part.id = offset
                    new_particles.append(part)
                    offset += 1

            particles.extend(new_particles)
            inter.particles = new_particles
            inter.particle_ids = np.array([part.id for part in new_particles])

        return {"reco_particles": particles}

    def check_merge(self, shower, track):
        """Check if a shower and a track can be merged.

        Parameters
        ----------
        shower : RecoParticle
            Shower particle to merge the track into
        track : RecoParticle
            Track particle that may be merged into the shower

        Returns
        -------
        bool
            Whether the track can be merged into the shower or not
        """
        # Check if the track is not too long to be merged
        if track.length > self.track_length_limit:
            return False

        # Check that the track dE/dx is not too large to be merged
        if self.track_dedx_limit is not None:
            dedx = cluster_dedx(track.points, track.depositions, track.start_point)
            if dedx > self.track_dedx_limit:
                return False

        # Check that the angular agreement between particles is sufficient
        angular_sep = abs(np.dot(track.start_dir, shower.start_dir))
        if angular_sep < self.angle_threshold:
            return False

        # Check that the distance between particles is not too large
        dist = cdist(shower.points, track.points).min()
        if dist > self.distance_threshold:
            return False

        # If all checks passed, return True
        return True


class ShowerStartCorrectionProcessor(PostBase):
    """Correct the start point of primary EM showers by finding the closest
    point to the vertex.

    This post-processor is used to patch upstream point proposal mistakes
    where the shower start point is not placed correctly.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "shower_start_correction"

    # Set of post-processors which must be run before this one is
    _upstream = ("direction",)

    def __init__(self, update_directions=True, radius=-1, optimize=True):
        """Store the shower start corrector parameters.

        Parameters
        ----------
        update_directions : bool, default True
            If `True`, the direction of showers for which the start point
            has been updated are recomputed with the updated start point
        radius : float, default -1
            Radius around the start voxel to include in the direction estimate
        optimize : bool, default True
            Optimize the number of points involved in the direction estimate
        """
        # Initialize the parent class
        super().__init__("interaction", "reco")

        # Store start corrector parameters
        self.update_directions = update_directions
        self.radius = radius
        self.optimize = optimize

    def process(self, data):
        """Update the shower start point using the closest point to the vertex.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for inter in data["reco_interactions"]:
            # Loop over the particles that make up the interaction
            for part in inter.particles:
                # If the particle PID is not a shower or not primary, skip
                if part.shape != SHOWR_SHP or not part.is_primary:
                    continue

                # Find the new start point of a shower
                new_start = self.correct_shower_start(inter, part)

                # If requested and needed, update the shower direction
                if self.update_directions and (new_start != part.start_point).any():
                    part.start_dir = cluster_direction(
                        part.points,
                        new_start,
                        max_dist=self.radius,
                        optimize=self.optimize,
                    )

                # Store the new start position
                part.start_point = new_start

    @staticmethod
    def correct_shower_start(inter, shower):
        """Function to correct the shower start point by finding the closest
        point to any primary track in the image.

        Parameters
        ----------
        inter : RecoInteraction
            Reco interaction to provide track points
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
