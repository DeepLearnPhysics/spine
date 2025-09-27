"""Cathode crossing identification + merging module."""

import numpy as np
from scipy.spatial.distance import cdist

from spine.data import RecoInteraction, TruthInteraction
from spine.math.distance import farthest_pair
from spine.post.base import PostBase
from spine.utils.geo import Geometry
from spine.utils.globals import TRACK_SHP
from spine.utils.gnn.cluster import cluster_direction

__all__ = ["CathodeCrosserProcessor"]


class CathodeCrosserProcessor(PostBase):
    """Find particles that cross the cathode of a LArTPC module that is divided
    into two TPCs. It might manifest itself into two forms:
    - If the particle is ~in-time, it will be a single particle, with
      potentially a small break/offset in the center
    - If the particle is sigificantly out-of-time, a cathode crosser will
      be composed of two distinct reconstructed particle objects
    """

    # Name of the post-processor (as specified in the configuration)
    name = "cathode_crosser"

    # Alternative allowed names of the post-processor
    aliases = ("find_cathode_crossers",)

    # Set of post-processors which must be run before this one is
    _upstream = ("direction",)

    # List of recognized offset methods
    _adjust_methods = ("distance", "projection", "dot")

    def __init__(
        self,
        crossing_point_tolerance,
        offset_tolerance,
        angle_tolerance,
        merge_crossers=True,
        adjust_crossers=True,
        adjust_method="distance",
        detector=None,
        geometry_file=None,
        run_mode="reco",
        truth_point_mode="points",
    ):
        """Initialize the cathode crosser finder algorithm.

        Parameters
        ----------
        crossing_point_tolerance : float
            Maximum allowed distance in the cathode plane (in cm) between two
            fragments of a cathode crosser to be considered compatible
        offset_tolerance
            Maximum allowed discrepancy between end-point to cathode offsets of
            two fragments of a cathode crosser to be considered compatible
        angle_tolerance : float
            Maximum allowed angle (in radians) between the directions of two
            fragments of a cathode crosser to be considered compatible
        merge_crossers : bool, default True
            If True, look for tracks that have been broken up at the cathode
            and merge them into one particle. Do the same for the interaction
            the crosser belongs to
        adjust_crossers : bool, default True
            If True, shifts cathode crosser positions to match at the cathode
        adjust_method : str, default distance
            Method used to adjust the cathode crossers:
            - 'distance': make the tracks meet at the cathode
            - 'projection': make the y-intercepts of the two tracks match in the
              xy and xz projection (in the least-squares sense)
            - 'dot': Make the dot-product of the two track directions match with
              the vector which joins the two end-points
        detector : str, optional
            Detector to get the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        """
        # Initialize the parent class
        super().__init__(("particle", "interaction"), run_mode, truth_point_mode)

        # Check on the moving method
        assert adjust_method in self._adjust_methods, (
            "Method used to adjust cathode crossers is not recognize: "
            f"{adjust_method}. It should be one of {self._adjust_methods}."
        )

        # Initialize the geometry
        self.geo = Geometry(detector, geometry_file)

        # Store the matching parameters
        self.crossing_point_tolerance = crossing_point_tolerance
        self.offset_tolerance = offset_tolerance
        self.angle_tolerance = angle_tolerance
        self.merge_crossers = merge_crossers
        self.adjust_crossers = adjust_crossers
        self.adjust_method = adjust_method

        # Add the points to the list of keys to load
        keys = {}
        if run_mode != "truth":
            keys["points"] = True
        if run_mode != "reco":
            keys[truth_point_key] = True
        self.update_keys(keys)

    def process(self, data):
        """Find cathode crossing particles in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Reset all particle/interaction cathode crossers
        for obj_key in self.obj_keys:
            for obj in data[obj_key]:
                obj.reset_cathode_crosser()

        # Reset all particle/interaction matches if any merging is to be done
        if self.merge_crossers:
            for obj_key in self.obj_keys:
                for obj in data[obj_key]:
                    obj.reset_match()

        # Loop over particle types
        update_dict = {}
        for part_key in self.particle_keys:
            # If there is no particle, nothing to do
            inter_key = part_key.replace("particle", "interaction")
            particles, interactions = data[part_key], data[inter_key]
            if len(particles) == 0:
                continue

            # Tag particles which live across multiple TPCs as crossers
            for part in particles:
                # Only track objects can be tagged as clear cathode crossers
                if part.shape != TRACK_SHP:
                    continue

                # Make sure the particle coordinates are expressed in cm
                self.check_units(part)

                # Make sure the sources are provided
                assert len(
                    self.get_sources(part)
                ), "Cannot identify cathode crossers without `sources`."

                # Check that the particle does not span more than a module
                # TODO: expand functionality to support multi-module crossers
                modules, tpcs = self.geo.get_contributors(self.get_sources(part))
                assert len(np.unique(modules)) == 1, (
                    "The cathode crosser identification/mergin post-"
                    "processor does not support multi-module crossers."
                )

                # A particle is a cathode crosser if there are two TPC sources
                # and points on either side of the cathode
                if len(np.unique(tpcs)) > 1:
                    points = self.get_points(part)
                    cpos = self.geo.tpc[modules[0]].cathode_pos
                    daxis = self.geo.tpc[modules[0]].drift_axis
                    if np.any(points[:, daxis] < cpos) and np.any(
                        points[:, daxis] > cpos
                    ):
                        part.is_cathode_crosser = len(np.unique(tpcs)) > 1

            # Merge particles which were split at the cathode, if requested.
            # Skip true particles which never need to be manually merged
            if self.merge_crossers and not particles[0].is_truth:
                assert particles[0].start_dir[0] > -np.inf, (
                    "Must reconstruct the direction of particles before "
                    "running the cathode crossing algorithm, abort."
                )

                particles, interactions = self.find_matches(particles)

            # Find the cathode offsets for cathode crossers
            for i, part in enumerate(particles):
                # Only bother to compute offsets for cathode crossers
                if not part.is_cathode_crosser:
                    continue

                # Store the displacement
                offset = self.get_cathode_offset(part)
                part.cathode_offset = offset

                # Adjust positions, if requested
                if self.adjust_crossers:
                    self.adjust_positions(
                        data, particles, interactions, part.id, offset
                    )

            # Update crossing interactions information
            for inter in interactions:
                crosser, offsets = False, []
                parts = [p for p in particles if p.interaction_id == inter.id]
                for part in parts:
                    if part.is_cathode_crosser:
                        crosser = True
                        offsets.append(part.cathode_offset)

                inter.is_cathode_crosser = crosser
                if crosser:
                    inter.cathode_offset = np.mean(offsets)

            # Update
            update_dict.update({part_key: particles})
            update_dict.update({inter_key: interactions})

        return update_dict

    def find_matches(self, particles):
        """Loop over all particles, find matches and update the
        particle/interaction list accordingly.

        Parameters
        ----------
        particles : List[RecoParticle]
            List of reconstructed particles

        Returns
        -------
        List[RecoParticle]
            Updated list of reconstructed particles
        List[RecoInteraction]
            Updated list of reconstructed interactions
        """
        # Restrict candidates to particles which are not already tagged and
        # that are track objects (only viable shape to merge)
        crossers = np.array([part.is_cathode_crosser for part in particles])
        tracks = np.array([part.shape == TRACK_SHP for part in particles])
        candidate_ids = np.where(~crossers & tracks)[0]

        # Try to find compatible tracks
        i = 0
        while i < len(candidate_ids):
            # Get the first particle and its properties
            ci = candidate_ids[i]
            pi = particles[ci]
            end_points_i = np.vstack([pi.start_point, pi.end_point])
            end_dirs_i = np.vstack([pi.start_dir, -pi.end_dir])

            # Check that the particle lives in a single TPC
            modules_i, tpcs_i = self.geo.get_contributors(self.get_sources(pi))
            if len(tpcs_i) != 1:
                i += 1
                continue

            # Get the cathode position, drift axis and cathode plane axes
            cpos = self.geo.tpc[modules_i[0]].cathode_pos
            daxis = self.geo.tpc[modules_i[0]].drift_axis
            caxes = np.array([i for i in range(3) if i != daxis])

            # Store the distance of the particle to the cathode
            tpc_offset = self.geo.get_min_volume_offset(
                end_points_i, modules_i[0], tpcs_i[0]
            )[daxis]

            # Loop over other tracks
            j = i + 1
            while j < len(candidate_ids):
                # Get the second particle object and its properties
                cj = candidate_ids[j]
                pj = particles[cj]
                end_points_j = np.vstack([pj.start_point, pj.end_point])
                end_dirs_j = np.vstack([pj.start_dir, -pj.end_dir])

                # Check that the particles live in TPCs of one module
                modules_j, tpcs_j = self.geo.get_contributors(self.get_sources(pj))
                if (
                    len(tpcs_j) != 1
                    or modules_i[0] != modules_j[0]
                    or tpcs_i[0] == tpcs_j[0]
                ):
                    j += 1
                    continue

                # Check if the two particles stop at roughly the same
                # position in the plane of the cathode
                compat = True
                dist_mat = cdist(end_points_i[:, caxes], end_points_j[:, caxes])
                argmin = np.argmin(dist_mat)
                pair_i, pair_j = np.unravel_index(argmin, (2, 2))
                compat &= dist_mat[pair_i, pair_j] < self.crossing_point_tolerance

                # Check if the offset of the two particles w.r.t. to the
                # cathode is compatible
                offset_i = end_points_i[pair_i, daxis] - cpos
                offset_j = end_points_j[pair_j, daxis] - cpos
                compat &= np.abs(offset_i + offset_j) < self.offset_tolerance

                # Check that the two directions where the two fragment
                # meet is consistent between the two
                cosang = np.dot(end_dirs_i[pair_i], -end_dirs_j[pair_j])
                compat &= np.arccos(cosang) < self.angle_tolerance

                # If compatible, merge
                if compat:
                    # Merge particles
                    self.merge_particles(particles, ci, cj)

                    # Update the candidate list to remove matched particle
                    candidate_ids[j:-1] = candidate_ids[j + 1 :] - 1
                    candidate_ids = candidate_ids[:-1]
                else:
                    j += 1

            # Increment
            i += 1

        # Update interaction list to reflect changes to the particle list
        interactions = []
        interaction_ids = np.array([p.interaction_id for p in particles])
        for i, inter_id in enumerate(np.unique(interaction_ids)):
            # Get particles in interaction inter_id
            particle_ids = np.where(interaction_ids == inter_id)[0]
            parts = [particles[i] for i in particle_ids]

            # Build interactions
            interaction = RecoInteraction.from_particles(parts)
            interaction.id = i

            # Reset the interaction ID of the constituent particles
            for j in particle_ids:
                particles[j].interaction_id = i

            # Append
            interactions.append(interaction)

        return particles, interactions

    def merge_particles(self, particles, idx_i, idx_j):
        """Given two particles which form a single cathode crosser, merge
        the two instances into one in place.

        It also takes can of changing the interaction IDs of sister particles
        and updating the interaction particle composition.

        Parameters
        ----------
        particles : List[RecoParticle]
            List of reconstructed particles
        idx_i : int
            Index of a cathode crosser (or a cathode crosser fragment)
        idx_j : int, optional
            Index of a matched cathode crosser fragment

        Returns
        -------
        List[RecoParticle]
            Updated list of reconstructed particles
        """
        # Merge particles, retain original interaction IDs
        inter_id_i = particles[idx_i].interaction_id
        inter_id_j = particles[idx_j].interaction_id
        particles[idx_i].merge(particles.pop(idx_j))
        particles[idx_i].is_cathode_crosser = True

        # Update the particle IDs and interaction IDs
        for i, part in enumerate(particles):
            part.id = i
            if part.interaction_id == inter_id_j:
                part.interaction_id = inter_id_i

        return particles

    def get_cathode_offset(self, particle):
        """Find the distance one must shift a particle points by to make
        both TPC contributions align at the cathode.

        Parameters
        ----------
        particle : Union[Particle, TruthParticle]
            Particle object

        Returns
        -------
        np.ndarray
            Offsets to apply to the each TPC contributions
        float
            Offset applied to the particle as a whole (signed by time)
        """
        # Get TPCs that contributed to this particle
        modules, tpcs = self.geo.get_contributors(self.get_sources(particle))
        assert (
            len(tpcs) == 2 and modules[0] == modules[1]
        ), "Can only handle particles crossing a single cathode."

        # Get the drift axis
        module = modules[0]
        daxis = self.geo.tpc[module].drift_axis

        # Loop over the contributing TPCs, find closest points/directions
        offsets = np.empty(2)
        closest_points = np.empty((2, 3))
        closest_dirs = np.empty((2, 3))
        for i, tpc in enumerate(tpcs):
            # Get the end points of the track segment
            index = self.geo.get_volume_index(self.get_sources(particle), module, tpc)
            points = self.get_points(particle)[index]
            idx0, idx1, _ = farthest_pair(points, "recursive")
            end_points = points[[idx0, idx1]]

            # Find the point closest to the cathode
            tpc_offset = self.geo.get_min_volume_offset(end_points, module, tpc)[daxis]
            cpos = self.geo.tpc[module][tpc].cathode_pos
            cdists = tpc_offset + cpos - end_points[:, daxis]
            argmin = np.argmin(np.abs(cdists))

            # Compute the offset to bring it to the cathode along the drift dir
            if self.adjust_method == "distance":
                dsign = self.geo.tpc[module][tpc].drift_sign
                offsets[i] = dsign * (cdists[argmin] - tpc_offset)
            else:
                closest_points[i] = end_points[argmin]
                closest_dirs[i] = -cluster_direction(points, closest_points[i])

        # Dispatch the offset estimation based on method
        if self.adjust_method == "distance":
            # Align the offsets to match the smallest of the two
            argmin = np.argmin(np.abs(offsets))
            global_offset = offsets[argmin]

        else:
            # Align based on angle, rather than distance to cathode
            if self.adjust_method == "projection":
                global_offset = self.projection_offset(closest_points, closest_dirs)
            else:
                global_offset = self.dot_offset(closest_points, closest_dirs)

            # Check which side of the cathode each TPC lives
            flip = (-1) ** (
                self.geo.tpc[module, tpcs[0]].boundaries[daxis].mean()
                > self.geo.tpc[module, tpcs[1]].boundaries[daxis].mean()
            )
            global_offset *= flip

        return global_offset

    @staticmethod
    def projection_offset(points, directions):
        """Finds the offset to optimize the alignment in xy and xz.

        It minimizes (xy intercept mismatch)^2 + (xz intercept mismatch)^2.

        Parameters
        ----------
        points : np.ndarray
            (2, 3) Set of two points (one per track fragment to merge)
        directions : np.ndarray
            (2, 3) Set of two directions (one per track fragment to merge)

        Returns
        -------
        float
            Optimal offset based on projection projection interecepts
        """
        # Check that the x component of the two vector is non-zero
        assert np.all(np.abs(directions[:, 0]) > 0.0), (
            "Cannot use the projection method if one the pieces of track "
            "is perfectly perpendicular to the x axis."
        )

        # Loop over the two projections
        diffs, weights = np.zeros(2), np.zeros(2)
        for proj in (1, 2):
            # Loop over the two track segments
            intercepts = np.empty(2)
            for idx in (0, 1):
                # Compute the y-intercept, store angle
                a = directions[idx, proj] / directions[idx, 0]
                intercepts[idx] = points[idx, proj] - a * points[idx, 0]
                weights[proj - 1] += a

            # Store the intercept offset between the two segments
            diffs[proj - 1] = intercepts[0] - intercepts[1]

        # Return
        return -np.sum(weights * diffs) / np.sum(weights**2)

    @staticmethod
    def dot_offset(points, directions):
        """Finds the offset to optimize the alignment between the line
        directions and the vector joining the two line points.

        It minimizes np.dot(norm, p2 - p1)

        Parameters
        ----------
        points : np.ndarray
            (2, 3) Set of two points (one per track fragment to merge)
        directions : np.ndarray
            (2, 3) Set of two directions (one per track fragment to merge)

        Returns
        -------
        float
            Optimal offset based on cross product
        """
        # Check that the x component of either vector is non-zero
        assert np.any(np.abs(directions[:, 0]) > 0.0), (
            "Cannot use the projection method if one the pieces of track "
            "is perfectly perpendicular to the x axis."
        )

        # Compute the norm of the two vectors
        norm = np.cross(directions[0], directions[1])

        # Compute the displacement vector
        v = points[1] - points[0]

        # Optimize delta such that the norm displacement vector are perpendicular
        return -np.dot(norm, v) / 2 * norm[0]

    def adjust_positions(self, data, particles, interactions, idx, offset):
        """Given a cathode crosser, apply the necessary position offsets to
        match it at the cathode.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        particles : List[ParticleBase]
            List of particles
        interactions : List[InteractionBase]
            List of interactions
        idx : int
            Index of a cathode crosser
        offset : float
            Offset to apply along the drift direction
        """
        # Fetch the appropriate point indexes to update
        truth = particles[idx].is_truth
        points_attr = "points" if not truth else self.truth_point_mode
        points_key = "points" if not truth else self.truth_point_key

        # Assign start and end point to a specific TPC
        closest_tpcs = {}
        for attr in ("start_point", "end_point"):
            key_point = getattr(particles[idx], attr)
            points = self.get_points(particles[idx])
            argmin = np.argmin(cdist(key_point[None, :], points))
            sources = self.get_sources(particles[idx])
            tpc_id = self.geo.get_contributors(sources[argmin][None, :])[1]
            closest_tpcs[attr] = tpc_id[0]

        # Get TPCs that contributed to this particle
        particle = particles[idx]
        modules, tpcs = self.geo.get_contributors(self.get_sources(particle))
        assert (
            len(tpcs) == 2 and modules[0] == modules[1]
        ), "Can only handle particles crossing a single cathode."

        # Get the particle's sisters
        inter_id = particle.interaction_id
        sisters = [p for p in particles if p.interaction_id == inter_id]

        # Get the drift axis
        m = modules[0]
        daxis = self.geo.tpc[m].drift_axis

        # Loop over contributing TPCs, shift the points in each independently
        for i, t in enumerate(tpcs):
            # Move each of the sister particles by the same amount
            offset_t = self.geo.tpc[m][t].drift_sign * offset
            for sister in sisters:
                # Find the index corresponding to the sister particle
                tpc_index = self.geo.get_volume_index(self.get_sources(sister), m, t)
                index = self.get_index(sister)[tpc_index]
                if not len(index):
                    continue

                # Update the sister position and the main position tensor
                self.get_points(sister)[tpc_index, daxis] += offset_t
                data[points_key][index, daxis] += offset_t

                # Update the start/end points appropriately
                if sister.id == idx:
                    for attr, closest_tpc in closest_tpcs.items():
                        if closest_tpc == t:
                            getattr(sister, attr)[daxis] += offset_t

                else:
                    sister.start_point[daxis] += offset_t
                    sister.end_point[daxis] += offset_t

        # Update the position attribute of the interaction
        points = [self.get_points(sister) for sister in sisters]
        setattr(interactions[inter_id], points_attr, np.vstack(points))
