"""Particle topology module."""

import numpy as np
from scipy.stats import pearsonr

from spine.math.decomposition import PCA
from spine.post.base import PostBase
from spine.utils.globals import ELEC_PID, PHOT_PID, SHOWR_SHP
from spine.utils.gnn.cluster import cluster_dedx, cluster_dedx_dir

__all__ = [
    "ParticleDEDXProcessor",
    "ParticleStartStraightnessProcessor",
    "ParticleSpreadProcessor",
]


class ParticleDEDXProcessor(PostBase):
    """Compute the dE/dx of the particle start by summing the energy depositions
    along the particle start and dividing by the total length of the start.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "local_dedx"

    # Aliases for the post-processor
    aliases = ("start_dedx", "end_dedx")

    # List of recognized dE/dx computation modes
    _modes = ("default", "direction")

    def __init__(
        self,
        radius=3.0,
        anchor=False,
        mode="direction",
        include_pids=(PHOT_PID, ELEC_PID),
        include_secondary=False,
    ):
        """Store the particle start dE/dx reconstruction parameters.

        Parameters
        ----------
        radius : float, default 3.0
            Radius around the start point to include in the dE/dx calculation.
        anchor : bool, default False
            If True, anchor the start point to the particle cluster. This
            is irrelevant for reconstruction particles (always anchored)
        mode : str, default 'direction'
            Method to use for dE/dx calculation.
        include_pids : list, default [0, 1]
            Particle species to compute the start dE/dx for
        include_secondary : bool, default False
            If `True`, computes the start dE/dx for secondary particles
        """
        # Initialize the parent class
        super().__init__("particle", "reco")

        # Check that the dE/dx mode is valid, store
        assert mode in self._modes, (
            f"dE/dx computation mode not recognized: {mode}. Should be "
            f"one of {self._modes}"
        )
        self.mode = mode

        # Store the dE/dx computation parameters
        self.radius = radius
        self.anchor = anchor
        self.include_pids = include_pids
        self.include_secondary = include_secondary

        # If the method involves the direction, must run the direction PP
        if mode == "direction":
            self.update_upstream("direction")

    def process(self, data):
        """Compute the start dE/dx for all particles in one entry.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco particles
        for part in data["reco_particles"]:
            # If the particle is a secondary and they are to be skipped, do
            if not self.include_secondary and not part.is_primary:
                continue

            # If the particle PID is not in the required list, skip
            if part.pid not in self.include_pids:
                continue

            # Loop over the two sides of the particle
            for side in ("start", "end"):
                # Showers have no end points, skip
                if side == "end" and part.shape == SHOWR_SHP:
                    continue

                # Fetch the point
                ref_point = getattr(part, f"{side}_point")

                # Compute the particle end dE/dx
                if self.mode == "default":
                    # Use all depositions within a radius of the particle point
                    dedx = cluster_dedx(
                        part.points,
                        part.depositions,
                        ref_point,
                        max_dist=self.radius,
                        anchor=self.anchor,
                    )

                else:
                    # Use the particle direction estimate as a guide
                    ref_dir = getattr(part, f"{side}_dir") * (-1) ** (end == "end")
                    dedx = cluster_dedx_dir(
                        part.points,
                        part.depositions,
                        ref_point,
                        ref_dir,
                        max_dist=self.radius,
                        anchor=self.anchor,
                    )[0]

                # Store the dE/dx
                setattr(part, f"{side}_dedx", dedx)


class ParticleStartStraightnessProcessor(PostBase):
    """Compute the relative straightness of the particle start.

    The start straightness is evaluated by computing the covariance matrix of
    the points in a neighborhood around the start of a particle and by
    estimating which fraction of that variance is explained by the first
    principal component of the decomposition.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "start_straightness"

    def __init__(
        self,
        radius=3.0,
        n_components=3,
        include_pids=(PHOT_PID, ELEC_PID),
        include_secondary=False,
    ):
        """Store the particle start straightness reconstruction parameters.

        Parameters
        ----------
        radius : float, default 3.0
            Radius around the start point to include in the straightness calculation.
        n_components : int, default 3
            Number of components to compute the PCA for
        include_pids : list, default [0, 1]
            Particle species to compute the start dE/dx for
        include_secondary : bool, default False
            If `True`, computes the start dE/dx for secondary particles
        """
        # Initialize the parent class
        super().__init__("particle", "reco")

        # Check that the number of components make sense, initialize PCA
        assert n_components > 0 and n_components <= 3, (
            "The number of PCA components should be at least 1 and at most "
            "3 (the dimensionality of 3D points)."
        )
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

        # Store the dE/dx computation parameters
        self.radius = radius
        self.include_pids = include_pids
        self.include_secondary = include_secondary

    def process(self, data):
        """Compute the start straightness for all particles in one entry.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco particles
        for part in data["reco_particles"]:
            # If the particle is a secondary and they are to be skipped, do
            if not self.include_secondary and not part.is_primary:
                continue

            # If the particle PID is not in the required list, skip
            if part.pid not in self.include_pids:
                continue

            # Compute the particle start straightness
            part.start_straightness = self.get_start_straightness(part)

    def get_start_straightness(self, particle):
        """Evaluates the straightness of the start of a particle by computing
        the PCA principal explained variance ratio.

        Parameters
        ----------
        particle : RecoParticle
            Reconstructed particle instance

        Returns
        -------
        float
            The first explained variance ratio of the PCA of the shower start
        """
        # Restrict the points to those around the start of the particle
        dists = np.linalg.norm(particle.points - particle.start_point, axis=1)
        points = particle.points[dists < self.radius]

        # Compute the explained variance ratio of the principal component
        if len(points) < self.n_components:
            return -1.0
        else:
            components, explained_variance = self.pca.fit(points)
            total_variance = np.sum(explained_variance)
            return explained_variance[0] / total_variance if total_variance > 0 else 0.0


class ParticleSpreadProcessor(PostBase):
    """Compute the directional and axial spreads of a particle.

    The directional spread is obtained by computing the mean direction and the
    weighted average cosine distance with respect to the mean direction.

    The axial spread is obtained by computing the pearson R correlation
    coefficient between the distance of the shower points from the startpoint
    along the shower axis and the perpendicular distance from the shower axis.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "particle_spread"

    # List of recognized start reference modes for spread evaluation
    _start_modes = ("vertex", "start_point")

    # Set of post-processors which must be run before this one is
    _upstream = ("direction",)

    def __init__(
        self,
        start_mode="vertex",
        length_scale=14.0,
        use_start_dir=False,
        include_pids=(PHOT_PID, ELEC_PID),
        include_secondary=False,
    ):
        """Store the particle spread reconstruction parameters.

        Parameters
        ----------
        start_mode : str, default 'vertex'
            Point used as a start point to evaluate the spread (can be one
            of 'vertex' or 'start_point')
        length_scale : float, default 14.0
            Length scale used to weight the directional spread estimator
        use_start_dir : bool, default False
            If `True`, use the direction estimate of the particle to compute
            the axial spread. If `False`, the direction is estimated using PCA
        include_pids : list, default [0, 1]
            Particle species to compute the start dE/dx for
        include_secondary : bool, default False
            If `True`, computes the start dE/dx for secondary particles
        """
        # Initialize the parent class
        super().__init__("interaction", "reco")

        # Check that the start point mode is valid, store
        assert start_mode in self._start_modes, (
            f"Spread start computation mode not recognized: {mode}. "
            f"Should be one of {self._modes}"
        )
        self.start_mode = start_mode

        # Store the spread computation parameters
        self.length_scale = length_scale
        self.use_start_dir = use_start_dir
        self.include_pids = include_pids
        self.include_secondary = include_secondary

        # If needed, initialize PCA
        if not use_start_dir:
            self.pca = PCA(n_components=3)

    def process(self, data):
        """Compute the shower spread for all particles in one entry.

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

                # If the particle PID is not in the required list, skip
                if part.pid not in self.include_pids:
                    continue

                # Fetch the appropriate reference point
                if self.start_mode == "vertex":
                    ref_point = inter.vertex
                else:
                    ref_point = part.start_point

                # Compute the spreads
                part.directional_spread = self.get_dir_spread(part.points, ref_point)
                part.axial_spread = self.get_axial_spread(
                    part.points, part.start_dir, ref_point
                )

    def get_dir_spread(self, points, ref_point):
        """Compute the directional spread of the particle by computing the mean
        direction and the weighted average cosine distance with respect to it.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Particle point coordinates
        ref_point : np.ndarray
            (3) Reference point

        Returns
        -------
        float
            Directional spread of the particle
        """
        # Compute the distance from the reference point to all particle points
        dists = np.linalg.norm(points - ref_point, axis=1)

        # Restrict points to those that do not coincide with the reference point
        index = np.where(dists > 0)[0]
        if len(index) == 0:
            return -1.0
        points, dists = points[index], dists[index]

        # Compute the exponential-weighted directional spread
        directions = (points - ref_point) / dists.reshape(-1, 1)
        weights = np.clip(np.exp(-dists / self.length_scale), min=1e-6)
        mean_direction = np.average(directions, weights=weights, axis=0)
        mean_direction /= np.linalg.norm(mean_direction)
        cosine = 1 - np.dot(directions, mean_direction)
        spread = np.average(cosine, weights=weights)

        return spread

    def get_axial_spread(self, points, start_dir, ref_point):
        """Compute the axial spread of a particle by evaluating the pearson R
        correlation coefficient between the distance of the particle points from
        the start point along the particle axis and the perpendicular distance
        from the particle axis.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Particle point coordinates
        start_dir : np.ndarray
            (3) Start direction of the particle
        ref_point : np.ndarray
            (3) Reference point

        Returns
        -------
        float
            Axial spread of the particle
        """
        # If there are less than 3 points, one cannot compute this quantity
        if len(points) < 3:
            return -np.inf

        # Fetch the particle direction
        if self.use_start_dir:
            v0 = start_dir
        else:
            components, _ = self.pca.fit(points)
            v0 = components[0]  # First principal component
            if np.dot(start_dir, v0) < 0.0:
                v0 *= -1

        # Compute the axial spread
        dists = np.linalg.norm(points - ref_point, axis=1)
        v = (ref_point - points) - (
            np.sum((ref_point - points) * v0, axis=1, keepdims=True)
            * np.broadcast_to(v0, points.shape)
        )
        perps = np.linalg.norm(v, axis=1)

        return pearsonr(dists, perps)[0]
