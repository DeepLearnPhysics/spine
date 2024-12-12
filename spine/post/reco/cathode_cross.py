"""Cathode crossing identification + merging module."""

import numpy as np
from scipy.spatial.distance import cdist

from spine.data import RecoInteraction, TruthInteraction

from spine.utils.globals import TRACK_SHP
from spine.utils.geo import Geometry
from spine.utils.numba_local import farthest_pair
from spine.utils.gnn.cluster import cluster_direction

from spine.post.base import PostBase

__all__ = ['CathodeCrosserProcessor']


class CathodeCrosserProcessor(PostBase):
    """Find particles that cross the cathode of a LArTPC module that is divided
    into two TPCs. It might manifest itself into two forms:
    - If the particle is ~in-time, it will be a single particle, with
      potentially a small break/offset in the center
    - If the particle is sigificantly out-of-time, a cathode crosser will
      be composed of two distinct reconstructed particle objects
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'cathode_crosser'

    # Alternative allowed names of the post-processor
    aliases = ('find_cathode_crossers',)

    def __init__(self, crossing_point_tolerance, offset_tolerance,
                 angle_tolerance, adjust_crossers=True, merge_crossers=True,
                 detector=None, geometry_file=None, run_mode='reco',
                 truth_point_mode='points'):
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
        adjust_crossers : bool, default True
            If True, shifts existing cathode crossers to fix the small breaks
            that may exist at the level of the cathode
        merge_crossers : bool, default True
            If True, look for tracks that have been broken up at the cathode
            and merge them into one particle
        detector : str, optional
            Detector to get the geometry from
        geometry_file : str, optional
            Path to a `.yaml` geometry file to load the geometry from
        """
        # Initialize the parent class
        super().__init__(
                ('particle', 'interaction'), run_mode, truth_point_mode)

        # Initialize the geometry
        self.geo = Geometry(detector, geometry_file)

        # Store the matching parameters
        self.crossing_point_tolerance = crossing_point_tolerance
        self.offset_tolerance = offset_tolerance
        self.angle_tolerance = angle_tolerance
        self.adjust_crossers = adjust_crossers
        self.merge_crossers = merge_crossers

        # Add the points to the list of keys to load
        keys = {}
        if run_mode != 'truth':
            keys['points'] = True
        if run_mode != 'reco':
            keys[truth_point_mode] = True

        self.update_keys(keys)

    def process(self, data):
        """Find cathode crossing particles in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle types
        update_dict = {}
        for part_key in self.particle_keys:
            # Check that the direction of particles in available
            if self.merge_crossers and len(data[part_key]) > 0:
                assert data[part_key][0].start_dir[0] > -np.inf, (
                        "Must reconstruct the direction of particles before "
                        "running the cathode crossing algorithm, abort.")

            # Find crossing particles already merged by the reconstruction
            prefix = part_key.split('_')[0]
            candidate_mask = np.zeros(len(data[part_key]), dtype=bool)
            for i, part in enumerate(data[part_key]):
                # Only bother to look for tracks that cross the cathode
                if part.shape != TRACK_SHP:
                    continue

                # Make sure the particle coordinates are expressed in cm
                self.check_units(part)

                # Get point coordinates, sources
                points = self.get_points(part)
                if not len(points):
                    continue
                assert len(self.get_sources(part)), (
                        "Cannot identify cathode crossers without `sources`.")

                # If the particle is composed of points from multiple
                # contributing TPCs in the same module, it is a cathode crosser
                modules, tpcs = self.geo.get_contributors(self.get_sources(part))
                part.is_cathode_crosser = (
                        len(np.unique(tpcs)) > len(np.unique(modules)))
                candidate_mask[i] = not part.is_cathode_crosser

                # Now measure the gap at the cathode, correct if requested
                # TODO: handle particles that cross a cathode in at least one
                # module but not all of them or cross multiple cathodes
                if (part.is_cathode_crosser and self.adjust_crossers and
                    len(tpcs) == 2):
                    # Adjust positions
                    self.adjust_positions(data, i)

            # If we do not want to merge broken crossers, our job here is done
            if not self.merge_crossers:
                continue

            # Try to find compatible tracks
            candidate_ids = np.where(candidate_mask)[0]
            i = 0
            while i < len(candidate_ids):
                # Get the first particle and its properties
                ci = candidate_ids[i]
                pi = data[part_key][ci]
                end_points_i = np.vstack([pi.start_point, pi.end_point])
                end_dirs_i = np.vstack([pi.start_dir, -pi.end_dir])

                # Check that the particle lives in one TPC
                modules_i, tpcs_i = self.geo.get_contributors(
                        self.get_sources(pi))
                if len(tpcs_i) != 1:
                    i += 1
                    continue

                # Get the cathode position, drift axis and cathode plane axes
                daxis = self.geo.tpc[modules_i[0]].drift_axis
                cpos = self.geo.tpc[modules_i[0]].cathode_pos
                caxes = np.array([i for i in range(3) if i != daxis])

                # Store the distance of the particle to the cathode
                tpc_offset = self.geo.get_min_volume_offset(
                        end_points_i, modules_i[0], tpcs_i[0])[daxis]
                cdists = end_points_i[:, daxis] - tpc_offset - cpos

                # Loop over other tracks
                j = i + 1
                while j < len(candidate_ids):

                    # Get the second particle object and its properties
                    cj = candidate_ids[j]
                    pj = data[part_key][cj]
                    end_points_j = np.vstack([pj.start_point, pj.end_point])
                    end_dirs_j = np.vstack([pj.start_dir, -pj.end_dir])

                    # Check that the particles live in TPCs of one module
                    modules_j, tpcs_j = self.geo.get_contributors(
                            self.get_sources(pj))

                    if (len(tpcs_j) != 1 or
                        modules_i[0] != modules_j[0] or
                        tpcs_i[0] == tpcs_j[0]):
                        j += 1
                        continue

                    # Check if the two particles stop at roughly the same
                    # position in the plane of the cathode
                    compat = True
                    dist_mat = cdist(
                            end_points_i[:, caxes], end_points_j[:, caxes])
                    argmin = np.argmin(dist_mat)
                    pair_i, pair_j = np.unravel_index(argmin, (2, 2))
                    compat &= (
                            dist_mat[pair_i, pair_j]
                            < self.crossing_point_tolerance)

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
                        # Merge particle and adjust positions
                        self.adjust_positions(data, ci, cj, truth=pi.is_truth)

                        # Update the candidate list to remove matched particle
                        candidate_ids[j:-1] = candidate_ids[j+1:] - 1
                        candidate_ids = candidate_ids[:-1]
                    else:
                        j += 1

                # Increment
                i += 1

            # Update crossing interactions information
            inter_key = f'{prefix}_interactions'
            for ia in data[inter_key]:
                crosser, offsets = False, []
                parts = [p for p in data[part_key] if p.interaction_id == ia.id]
                for p in parts:
                    if p.interaction_id != ia.id:
                        continue
                    crosser |= p.is_cathode_crosser
                    if p.is_cathode_crosser:
                        offsets.append(p.cathode_offset)

                if crosser:
                    ia.is_cathode_crosser = crosser
                    ia.cathode_offset = np.mean(offsets)

            # Update
            update_dict.update({part_key: data[part_key]})
            update_dict.update({inter_key: data[inter_key]})

        return update_dict

    def adjust_positions(self, data, idx_i, idx_j=None, truth=False):
        """Given a cathode crosser (either in one or two pieces), apply the
        necessary position offsets to match it at the cathode.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        idx_i : int
            Index of a cathode crosser (or a cathode crosser fragment)
        idx_j : int, optional
            Index of a matched cathode crosser fragment
        truth : bool, default False
            If True, adjust truth object positions

        Results
        -------
        np.ndarray
           (N, 3) Point coordinates
        """
        # If there are two indexes, create a new merged particle object
        prefix = 'truth' if truth else 'reco'
        part_key, inter_key = f'{prefix}_particles', f'{prefix}_interactions'
        points_attr = 'points' if not truth else self.truth_point_mode
        points_key = 'points' if not truth else self.truth_point_key
        particles = data[part_key]
        if idx_j is not None:
            # Merge particles
            int_id_i = particles[idx_i].interaction_id
            int_id_j = particles[idx_j].interaction_id
            particles[idx_i].merge(particles.pop(idx_j))

            # Update the particle IDs and interaction IDs
            assert idx_j > idx_i
            for i, p in enumerate(particles):
                p.id = i
                if p.interaction_id == int_id_j:
                    p.interaction_id = int_id_i

        # Assign start and end point to a specific TPC
        closest_tpcs = {}
        for attr in ('start_point', 'end_point'):
            key_point = getattr(particles[idx_i], attr)
            points = self.get_points(particles[idx_i])
            argmin = np.argmin(cdist(key_point[None, :], points))
            sources = self.get_sources(particles[idx_i])
            tpc_id = self.geo.get_contributors(sources[argmin][None, :])[1]
            closest_tpcs[attr] = tpc_id[0]

        # Get TPCs that contributed to this particle
        particle = particles[idx_i]
        modules, tpcs = self.geo.get_contributors(self.get_sources(particle))
        assert len(tpcs) == 2 and modules[0] == modules[1], (
                "Can only handle particles crossing a single cathode.")

        # Get the particle's sisters
        int_id = particle.interaction_id
        sisters = [p for p in particles if p.interaction_id == int_id]

        # Get the cathode position
        m = modules[0]
        daxis = self.geo.tpc[m].drift_axis
        cpos = self.geo.tpc[m].cathode_pos

        # Loop over contributing TPCs, shift the points in each independently
        offsets, global_offset = self.get_cathode_offsets(
                particle, m, tpcs)
        for i, t in enumerate(tpcs):
            # Move each of the sister particles by the same amount
            for sister in sisters:
                # Find the index corresponding to the sister particle
                tpc_index = self.geo.get_volume_index(
                        self.get_sources(sister), m, t)
                index = self.get_index(sister)[tpc_index]
                if not len(index):
                    continue

                # Update the sister position and the main position tensor
                self.get_points(sister)[tpc_index, daxis] -= offsets[i]
                data[points_key][index, daxis] -= offsets[i]

                # Update the start/end points appropriately
                if sister.id == idx_i:
                    for attr, closest_tpc in closest_tpcs.items():
                        if closest_tpc == t:
                            getattr(sister, attr)[daxis] -= offsets[i]

                else:
                    sister.start_point[daxis] -= offsets[i]
                    sister.end_point[daxis] -= offsets[i]

        # Store crosser information
        particle.is_cathode_crosser = True
        particle.cathode_offset = global_offset

        # Update interactions, if need be
        if idx_j is None:
            # In this case, just need to update the positions
            interactions = data[inter_key]
            points = [self.get_points(sister) for sister in sisters]
            setattr(interactions[int_id], points_attr, np.vstack(points))

        else:
            # Merge newly formed particles into interactions
            interactions = []
            interaction_ids = np.array([p.interaction_id for p in particles])
            for i, int_id in enumerate(np.unique(interaction_ids)):
                # Get particles in interaction int_id
                particle_ids = np.where(interaction_ids == int_id)[0]
                parts = [particles[i] for i in particle_ids]

                # Build interactions
                if not truth:
                    interaction = RecoInteraction.from_particles(parts)
                    interaction.id = i
                else:
                    interaction = TruthInteraction.from_particles(parts)
                    interaction.id = i
                    interaction.truth_id = int_id

                # Reset the interaction ID of the constiuent particles
                for j in particle_ids:
                    particles[j].interaction_id = i

                # Append
                interactions.append(interaction)

            data[inter_key] = interactions

    def get_cathode_offsets(self, particle, module, tpcs):
        """Find the distance one must shift a particle points by to make
        both TPC contributions align at the cathode.

        Parameters
        ----------
        particle : Union[Particle, TruthParticle]
            Particle object
        module : int
            Module ID
        tpcs : List[int]
            List of TPC IDs

        Returns
        -------
        np.ndarray
            Offsets to apply to the each TPC contributions
        float
            General offset for this particle (proxy of out-of-time displacement)
        """
        # Get the cathode position
        daxis = self.geo.tpc[module].drift_axis
        cpos = self.geo.tpc[module].cathode_pos
        dvector = (np.arange(3) == daxis).astype(float)

        # Check which side of the cathode each TPC lives
        flip = (-1) ** (
                self.geo.tpc[module, tpcs[0]].boundaries[daxis].mean()
                > self.geo.tpc[module, tpcs[1]].boundaries[daxis].mean())

        # Loop over the contributing TPCs
        closest_points = np.empty((2, 3))
        offsets = np.empty(2)
        for i, t in enumerate(tpcs):
            # Get the end points of the track segment
            index  = self.geo.get_volume_index(
                    self.get_sources(particle), module, t)
            points = self.get_points(particle)[index]
            idx0, idx1, _ = farthest_pair(points, 'recursive')
            end_points = points[[idx0, idx1]]

            # Find the point closest to the cathode
            tpc_offset = self.geo.get_min_volume_offset(
                end_points, module, t)[daxis]
            cdists = end_points[:, daxis] - tpc_offset - cpos
            argmin = np.argmin(np.abs(cdists))
            closest_points[i] = end_points[argmin]

            # Compute the offset to bring it to the cathode
            offsets[i] = cdists[argmin] + tpc_offset

        # Now optimize the offsets based on angular matching
        # xing_point = np.mean(closest_points, axis=0)
        # xing_point[daxis] = cpos
        # for i, t in enumerate(tpcs):
        #     end_dir = -cluster_direction(points, closest_points[i])
        #     factor = (cpos - closest_points[i, daxis])/end_dir[daxis]
        #     intersection = closest_points[i] + factor * end_dir
        #     vplane = dvector - end_dir/end_dir[daxis] if end_dir[daxis] else -end_dir
        #     dplane = intersection - xing_point
        #     disp = np.dot(dplane, vplane)/np.dot(vplane, vplane)
        #     offsets[i] = [disp, offsets[i]][np.argmin(np.abs([disp, offsets[i]]))]

        # Take the average offset as the value to use
        global_offset = flip * (offsets[1] - offsets[0])/2.

        return offsets, global_offset
