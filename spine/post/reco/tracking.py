"""Tracking reconstruction modules."""

import numpy as np

from scipy.spatial.distance import cdist

from spine.utils.globals import (
        SHOWR_SHP, TRACK_SHP, MUON_PID, PION_PID, PROT_PID, KAON_PID)
from spine.utils.energy_loss import csda_table_spline
from spine.utils.gnn.cluster import cluster_dedx
from spine.utils.tracking import get_track_length
from spine.post.base import PostBase

__all__ = ['CSDAEnergyProcessor', 'TrackValidityProcessor',
           'TrackShowerMergerProcessor']


class CSDAEnergyProcessor(PostBase):
    """Reconstruct the kinetic energy of tracks based on their range in liquid
    argon using the continuous slowing down approximation (CSDA).
    """
    name = 'csda_ke'
    aliases = ['reconstruct_csda_energy']

    def __init__(self, tracking_mode='step_next',
                 include_pids=[MUON_PID, PION_PID, PROT_PID, KAON_PID],
                 obj_type='particle', run_mode='both',
                 truth_point_mode='points', **kwargs):
        """Store the necessary attributes to do CSDA range-based estimation.

        Parameters
        ----------
        tracking_mode : str, default 'step_next'
            Method used to compute the track length (one of 'displacement',
            'step', 'step_next', 'bin_pca' or 'spline')
        include_pids : list, default [2, 3, 4, 5]
            Particle species to compute the kinetic energy for
        **kwargs : dict, optional
            Additional arguments to pass to the tracking algorithm
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode)

        # Fetch the functions that map the range to a KE
        self.include_pids = include_pids
        self.splines = {
                ptype: csda_table_spline(ptype) for ptype in include_pids}

        # Store the tracking parameters
        self.tracking_mode = tracking_mode
        self.tracking_kwargs = kwargs

    def process(self, data):
        """Reconstruct the CSDA KE estimates for each particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle objects
        for k in self.fragment_keys + self.particle_keys:
            for obj in data[k]:
                # Only run this algorithm on tracks that have a CSDA table
                if not ((obj.shape == TRACK_SHP) and
                        (obj.pid in self.include_pids)):
                    continue

                # Make sure the object coordinates are expressed in cm
                self.check_units(obj)

                # Get point coordinates
                points = self.get_points(obj)
                if not len(points):
                    continue

                # Compute the length of the track
                length = get_track_length(
                        points, point=obj.start_point,
                        method=self.tracking_mode, **self.tracking_kwargs)

                # Store the length
                if not obj.is_truth:
                    obj.length = length
                else:
                    obj.reco_length = length

                # Compute the CSDA kinetic energy
                if length > 0.:
                    obj.csda_ke = self.splines[obj.pid](length).item()
                else:
                    obj.csda_ke = 0.


class TrackValidityProcessor(PostBase):
    """Check if track is valid based on the proximity to reconstructed vertex.
    Can also reject small tracks that are close to the vertex (optional).
    """
    name = 'track_validity'
    aliases = ['track_validity_processor']

    def __init__(self, threshold=3., ke_threshold=50.,
                 check_small_track=False, **kwargs):
        """Initialize the track validity post-processor.

        Parameters
        ----------
        threshold : float, default 3.0
            Vertex distance above which a track is not considered a primary
        ke_theshold : float, default 50.0
            Kinetic energy threshold below which a track close to the vertex
            is deemed not primary/not valid
        check_small_track : bool, default False
            Whether or not to apply the small track KE cut
        """
        # Initialize the parent class
        super().__init__('interaction', 'reco')

        self.threshold = threshold
        self.ke_threshold = ke_threshold
        self.check_small_track = check_small_track

    def process(self, data):
        """Loop through reco interactions and modify reco particle's
        primary label based on the proximity to reconstructed vertex.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle objects
        for ia in data['reco_interactions']:
            for p in ia.particles:
                if p.shape == TRACK_SHP and p.is_primary:
                    # Check vertex attachment
                    dist = np.linalg.norm(p.points - ia.vertex, axis=1)
                    p.vertex_distance = dist.min()
                    if p.vertex_distance > self.threshold:
                        p.is_primary = False
                    # Check if track is small and within radius from vertex
                    if self.check_small_track:
                        if ((dist < self.threshold).all()
                            and p.ke < self.ke_threshold):
                            p.is_valid = False
                            p.is_primary = False


class TrackShowerMergerProcessor(PostBase):
    """Merge tracks into showers based on a set of selection criteria.
    """
    name = 'merge_track_to_shower'
    aliases = ['track_shower_merger']

    def __init__(self, angle_threshold=10, adjacency_threshold=0.5,
                 dedx_threshold=-1, track_length_limit=50, **kwargs):
        """Post-processor to merge tracks into showers.

        Parameters
        ----------
        angle_threshold : float, default 0.95
            Check if track and shower cosine similarity is greater than this value.
        adjacency_threshold : float, default 0.5
            Check if track and shower is within this threshold distance.
        dedx_limit : int, default -1
            Check if the track dedx is below this value,
            to avoid merging protons.
        track_length_limit : int, default 40
            Check if track length is below this value,
            to avoid merging long tracks.
        """
        # Initialize the parent class
        super().__init__('interaction', 'reco')

        self.angle_threshold = abs(np.cos(np.radians(angle_threshold)))
        self.adjacency_threshold = adjacency_threshold
        self.dedx_threshold = dedx_threshold
        self.track_length_limit = track_length_limit

    def process(self, data):
        """Loop over the reco interactions and merge tracks into showers,
        if they pass the selection criteria.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the reco interactions
        interactions = []
        for ia in data['reco_interactions']:
            # Leading shower and its ke
            shower_p = None
            shower_p_max_ke = 0
            # Loop over particles, select the ones that pass a threshold
            for p in ia.particles:
                if p.is_primary and p.shape == SHOWR_SHP:
                    if p.ke > shower_p_max_ke:
                        shower_p = p
                        shower_p_max_ke = p.ke
            if shower_p is None:
                interactions.append(ia)
                continue
            new_particles = []
            for p in ia.particles:
                if p.shape == TRACK_SHP and p.is_primary:
                    should_merge = check_merge(p, shower_p,
                        angle_threshold=self.angle_threshold,
                        adjacency_threshold=self.adjacency_threshold,
                        dedx_limit=self.dedx_threshold,
                        track_length_limit=self.track_length_limit)
                    if should_merge:
                        merge_track_to_shower(shower_p, p)
                        p.is_valid = False
                    else:
                        new_particles.append(p)
                else:
                    new_particles.append(p)
            if len(ia.particles) != len(new_particles):
                ia.particles = new_particles
            interactions.append(ia)

        data['reco_interactions'] = interactions


def merge_track_to_shower(p1, p2):
    """Merge a track p2 into shower p1.

    Parameters
    ----------
    p1 : RecoParticle
        Shower particle to merge p1 into.
    p2 : RecoParticle
        Track particle p2 that will be merged into p1.
    """
    # Sanity checks
    assert p1.shape == SHOWR_SHP
    assert p2.shape == TRACK_SHP

    # Stack the two particle array attributes together
    for attr in ['index', 'depositions']:
        val = np.concatenate([getattr(p1, attr), getattr(p2, attr)])
        setattr(p1, attr, val)
    for attr in ['points', 'sources']:
        val = np.vstack([getattr(p1, attr), getattr(p2, attr)])
        setattr(p1, attr, val)

    # Select track startpoint as new startpoint
    p1.start_point = np.copy(p2.start_point)
    p1.calo_ke = p1.calo_ke + p2.ke

    # If one of the two particles is a primary, the new one is
    p1.is_primary = max(p1.is_primary, p2.is_primary)
    if p2.primary_scores[-1] > p1.primary_scores[-1]:
        p1.primary_scores = p2.primary_scores


def check_merge(p_track, p_shower, angle_threshold=0.95,
                adjacency_threshold=0.5, dedx_limit=-1, track_length_limit=40):
    """Check if a track and a shower can be merged.

    Parameters
    ----------
    p_track : RecoParticle
        Track particle that will be merged into the shower.
    p_shower : RecoParticle
        Shower particle to merge the track into.
    angle_threshold : float, default 0.95
        Check if track and shower cosine distance is greater than this value.
    adjacency_threshold : float, default 0.5
        Check if track and shower is within threshold distance.
    dedx_limit : int, default -1
        Check if the track dedx is below this value,
        to avoid merging protons.
    track_length_limit : int, default 40
        Check if track length is below this value,
        to avoid merging long tracks.

    Returns
    -------
    result : bool
        True if the track and shower can be merged, False otherwise.
    """

    check_direction = False
    check_adjacency = False
    check_dedx = True
    check_track_energy = False

    angular_sep = abs(np.sum(p_track.start_dir * p_shower.start_dir))

    if angular_sep > angle_threshold:
        check_direction = True

    if cdist(p_shower.points.reshape(-1, 3),
             p_track.points.reshape(-1, 3)).min() < adjacency_threshold:
        check_adjacency = True

    dedx = cluster_dedx(p_track.points, p_track.depositions, p_track.start_point)
    if dedx > dedx_limit:
        check_dedx = False
    if p_track.length < track_length_limit:
        check_track_energy = True

    result = (check_dedx and check_direction and
              check_adjacency and check_track_energy)

    return result
