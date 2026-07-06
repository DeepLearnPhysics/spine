"""Module that supports TPC-CRT matching."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from spine.constants import TRACK_SHP
from spine.geo import GeoManager
from spine.math.distance import cdist


class CRTMatcher:
    """Matches particles and CRT hits by matching the propagated TPC tracks
    with the position of CRT hits.
    """

    # List of valid matching methods
    _match_methods = ("threshold", "best")

    def __init__(
        self,
        driftv: float,
        match_method: str = "threshold",
        tolerance: float = 5.0,
        time_window: Sequence[float] | None = None,
        min_part_size: int | None = None,
        min_crt_pe: float | None = None,
        match_distance: float | None = None,
    ) -> None:
        """Initialize the CRT/TPC matcher.

        Parameters
        ----------
        driftv : float
            Drift velocity in cm/us
        match_method : str, default 'threshold'
            Matching method (one of 'threshold' or 'best')
            - 'threshold': If a TPC track can be propagated such that it matches
            the CRT hit within some threshold, they are matched
            - 'best': Only allow at most a single TPC match per CRT hit
        tolerance : float, default 5
            Distance along x in cm that a track is allowed to float around the
            expected position given a CRT hit time
        time_window : Sequence[float], optional
            List of [min, max] values of optical flash times to consider
        min_part_size : int, optional
            Minimum number of voxel in an particle to consider it
        min_crt_pe : float, optional
            Minimum number of PE in a CRT hit to consider it
        match_distance : float, optional
            If a threshold is used, specifies the acceptable distance
        """
        # Check validity of key parameters
        if match_method not in self._match_methods:
            raise ValueError(
                f"CRT matching method not recognized: {match_method}. "
                f"Must be one of {self._match_methods}."
            )

        if match_method == "threshold":
            assert match_distance is not None, (
                "When using the `threshold` method, must specify the "
                "`match_distance` parameter."
            )
            self.match_distance = float(match_distance)

        # Store the geometry manager and drift velocity
        self.geo = GeoManager.get_instance()
        self.driftv = driftv

        # Store the flash matching parameters
        self.match_method = match_method
        self.tolerance = tolerance
        self.time_window = time_window
        self.min_part_size = min_part_size
        self.min_crt_pe = min_crt_pe

    def get_matches(
        self, particles: Sequence[Any], crthits: Sequence[Any]
    ) -> list[tuple[Any, Any, float]]:
        """Makes [particle, crthit] pairs that are compatible.

        Parameters
        ----------
        particles : Sequence[RecoParticle | TruthParticle]
            List of particles
        crthits : Sequence[CRTHit]
            List of CRT hits

        Returns
        -------
        list[tuple[Particle, CRTHit, float]]
            List of [particle, CRT hit, distance] triplets
        """
        # Check on the geometry
        assert (
            self.geo.crt is not None
        ), "Cannot find CRT-TPC matches without `crt` geometry."

        # Restrict the CRT hits and particles based on the configuration
        particles, crthits = self.restrict_objects(particles, crthits)
        if len(crthits) == 0 or len(particles) == 0:
            return []

        # For each track, determine how much it is allowed to move along the drift axis
        fit_ranges = np.empty((len(particles), 2), dtype=np.float32)
        for i, part in enumerate(particles):
            # Make sure the sources are provided
            # TODO: change the way points/sources are fetched
            # points = self.get_points(part)
            # sources = self.get_sources(part)
            points, sources = part.points, part.sources
            assert len(sources), "Cannot find CRT-TPC matches without `sources`."

            # Check the TPC composition of the object
            modules, tpcs = self.geo.get_contributors(sources)
            if len(tpcs) == 1:
                # If the particle is made up of a single TPC, it can be
                # placed anywhere within the boundaries of that TPC.
                # Fetch the TPC boundaries along the drift axis
                chamber = self.geo.tpc[modules[0]][tpcs[0]]
                dsign, daxis = chamber.drift_sign, chamber.drift_axis
                apos, cpos = chamber.anode_pos, chamber.cathode_pos
                llim, ulim = (cpos, apos) if dsign > 0 else (apos, cpos)

                # Create boundary conditions
                pmin, pmax = np.min(points[:, daxis]), np.max(points[:, daxis])
                lbound, ubound = llim - pmin, ulim - pmax
                if dsign > 0:
                    fit_ranges[i] = [lbound, ubound]
                else:
                    fit_ranges[i] = [-ubound, -lbound]

            else:
                # If the particle is made up of multiple TPCs, it is a cathode
                # crosser which only has a single allowed position
                fit_ranges[i] = [0.0, 0.0]  # TODO

        # Find matches, start by looping over CRT hits
        matches = []
        for hit in crthits:
            # Fetch the normal to the CRT plane (self.geo.crt guaranteed non-None by __init__ assert)
            crt_plane = self.geo.crt.get_plane(hit.position, hit.plane)
            normal = crt_plane.normal

            # Downselect to particles which can match the time of the hit
            offset = hit.time * self.driftv
            mask = (offset >= fit_ranges[:, 0] - self.tolerance) & (
                offset <= fit_ranges[:, 1] + self.tolerance
            )
            part_ids = np.where(mask)[0]

            # Loop over candidate particles
            for part_id in part_ids:
                # Get the point and direction closest to the CRT hit
                part = particles[part_id]
                points = np.vstack((part.start_point, part.end_point))
                dirs = np.vstack((part.start_dir, part.end_dir))
                closest_id = np.argmin(np.sum((points - hit.center) ** 2, axis=1))
                point, direction = np.copy(points[closest_id]), dirs[closest_id]

                # Move the point according to the expected offset
                # TODO: change the way points/sources are fetched
                closest_id = np.argmin(cdist(part.points, point[None, :]))
                modules, tpcs = self.geo.get_contributors(
                    part.sources[closest_id][None, :]
                )
                chamber = self.geo.tpc[modules[0]][tpcs[0]]
                dsign, daxis = chamber.drift_sign, chamber.drift_axis
                point[daxis] += dsign * offset

                # Check how close they meet in the plane of the CRT
                # TODO: integrate uncertainties properly
                intercept = self.line_plane_intercept(
                    point, direction, hit.center, normal
                )
                dist = np.linalg.norm(intercept - hit.center)
                if dist < self.match_distance:
                    matches.append((part, hit, dist))

        return matches

    def restrict_objects(
        self, particles: Sequence[Any], crthits: Sequence[Any]
    ) -> tuple[list[Any], list[Any]]:
        """Restrict the list of particles/CRT hits to match the configuration.

        Parameters
        ----------
        particles : Sequence[RecoParticle | TruthParticle]
            List of particles
        crthits : Sequence[CRTHit]
            List of CRT hits

        Returns
        -------
        list[RecoParticle | TruthParticle]
            Restricted list of particles
        list[CRTHit]
            Restricted list of CRT hits
        """
        # Restrict the CRT hits to those that fit the selection criterion
        if self.time_window is not None:
            t1, t2 = self.time_window
            crthits = [h for h in crthits if (h.time > t1 and h.time < t2)]
        else:
            crthits = list(crthits)

        # Restrict the particles to tracks that fit the selection criteria
        particles = [part for part in particles if part.shape == TRACK_SHP]
        if self.min_part_size is not None:
            particles = [part for part in particles if part.size > self.min_part_size]

        return particles, crthits

    def line_plane_intercept(
        self,
        line_p: np.ndarray,
        line_v: np.ndarray,
        plane_p: np.ndarray,
        plane_n: np.ndarray,
    ) -> np.ndarray:
        """Find the intercept between a line and a plane in 3D.

        Parameters
        ----------
        line_p : np.ndarray
            (3) Position of a point on the line
        line_v : np.ndarray
            (3) Direction vector of the line
        plane_p : np.ndarray
            (3) Position of a point on the plane
        plane_n : np.ndarray
            (3) Normal to the plane

        Returns
        -------
        np.ndarray
            (3) Position of the intercept
        """
        # Compute dot product, check that line/plane are not parallel
        denom = np.dot(line_v, plane_n)
        if denom == 0.0:
            # Line parallel to plane
            return np.full(3, -np.inf)

        # Compute intercept
        t = np.dot(plane_p - line_p, plane_n) / denom

        return line_p + t * line_v
