import numpy as np

from spine.utils.globals import (PHOT_PID, PROT_PID, PION_PID, ELEC_PID, 
                                 SHOWR_SHP, TRACK_SHP)

from spine.post.base import PostBase
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from spine.utils.tracking import get_track_segment_dedxs

from spine.utils.numba_local import cdist

from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN

__all__ = ['ConversionDistanceProcessor', 
           'ShowerStartpointCorrectionProcessor', 
           'MichelTaggingProcessor']


class ConversionDistanceProcessor(PostBase):
    """Enforce additional constraint on valid primary electron showers
    using vertex-to-shower separation distance. 
    
    NOTE: This processor can only change reco electron shower pid to
    photon pid depending on the distance threshold. 
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'shower_conversion_distance'

    # Alternative allowed names of the post-processor
    aliases = ('shower_separation_processor',)
    
    def __init__(self, threshold=-1.0, vertex_mode='vertex_points', 
                 inplace=True, eps=0.6, primary_override_ke=None):
        """Specify the EM shower conversion distance threshold and
        the type of vertex to use for the distance calculation.

        Parameters
        ----------
        threshold : float, default -1.0
            If EM shower has a conversion distance greater than this,
            its PID will be changed to PHOT_PID.
        vertex_mode : str, default 'vertex'
            The type of vertex to use for the distance calculation.
            'protons': Distance between the shower startpoint and the
            closest proton/pion point.
            'vertex_points': Distance between the vertex and all shower points.
            'vertex_startpoint': Distance between the vertex and the predicted
            shower startpoint.
        """
        super().__init__('interaction', 'reco')
        
        self.threshold = threshold
        self.vertex_mode = vertex_mode
        self.inplace = inplace
        self.eps = eps
        if primary_override_ke is None:
            self.primary_override_ke = np.inf
        else:
            self.primary_override_ke = primary_override_ke
        
    def process(self, data):
        """Update reco interaction topologies using the conversion
        distance cut.

        Parameters
        ----------
        data : dict
            Dictionaries of data products

        Raises
        ------
        ValueError
            If provided vertex mode is invalid.
        """
        # Loop over the reco interactions
        for ia in data['reco_interactions']:
            criterion = -np.inf
            criterion_alt = -np.inf
            
            leading_shower, energy = None, -np.inf
            
            for p in ia.particles:
                
                if p.shape == SHOWR_SHP and p.ke > self.primary_override_ke:
                    # Check if shower is touching vertex
                    points = p.points
                    dists = np.linalg.norm(points - ia.vertex, axis=1)
                    if np.min(dists) < self.threshold:
                        p.is_primary = True
                        p.is_valid = True
                
                if (p.shape == SHOWR_SHP and p.is_primary):
                    
                    if self.vertex_mode == 'protons':
                        criterion = self.convdist_protons(ia, p)
                    elif self.vertex_mode == 'vertex_points':
                        criterion = self.convdist_vertex_points(ia, p)
                    elif self.vertex_mode == 'vertex_startpoint':
                        criterion = self.convdist_vertex_startpoint(ia, p)
                    # elif self.vertex_mode == 'vertex_relaxed':
                    else:
                        raise ValueError('Invalid point mode')
                    
                    p.vertex_distance = criterion
                    
                    if p.ke > energy:
                        leading_shower = p
                        energy = p.ke
                    
                    if p.pid == ELEC_PID:
                        
                        if self.inplace:
                            if criterion >= self.threshold:
                                p.pid = PHOT_PID
                            
            if leading_shower is None:
                ia.leading_shower_vertex_distance = -np.inf
            else:
                ia.leading_shower_vertex_distance = leading_shower.vertex_distance
            
    @staticmethod        
    def convdist_protons(ia, shower_p):
        """Helper function to compute the distance between the shower
        startpoint and the closest proton point.

        Parameters
        ----------
        ia : RecoInteraction
            Reco interaction to apply the conversion distance cut.
        shower_p : RecoParticle
            Member particle of the interaction, assumed to be the primary
            electron/gamma shower.

        Returns
        -------
        start_to_closest_proton : float
            Closest distance between the shower startpoint 
            and proton/pion points.
        """
        start_to_closest_proton = -np.inf
        for p2 in ia.particles:
            if (p2.pid == PROT_PID or p2.pid == PION_PID) and p2.is_primary:
                proton_points.append(p2.start_point)
        if len(proton_points) > 0:
            proton_points = np.vstack(proton_points)
            start_to_closest_proton = cdist(proton_points, shower_p.points)
            start_to_closest_proton = start_to_closest_proton.min()
        else:
            start_to_closest_proton = -np.inf
        return start_to_closest_proton

    @staticmethod
    def convdist_vertex_points(ia, shower_p, mode='vertex'):
        """Helper function to compute the closest distance 
        between the vertex and all shower points. 

        Parameters
        ----------
        ia : RecoInteraction
            Reco interaction to apply the conversion distance cut.
        shower_p : RecoParticle
            Member particle of the interaction, assumed to be the primary
            electron/gamma shower.

        Returns
        -------
        start_to_closest_proton : float
            Closest distance between the shower startpoint and proton points.
        """
        vertex_dist = -np.inf
        vertex = getattr(ia, mode)
        vertex_dist = cdist(vertex.reshape(1, -1), shower_p.points)
        vertex_dist = vertex_dist.min()
        return vertex_dist
    
    @staticmethod
    def convdist_vertex_startpoint(ia, shower_p):
        """Helper function to compute the closest distance 
        between the vertex and predicted shower startpoint. 

        Parameters
        ----------
        ia : RecoInteraction
            Reco interaction to apply the conversion distance cut.
        shower_p : RecoParticle
            Member particle of the interaction, assumed to be the primary
            electron/gamma shower.

        Returns
        -------
        start_to_closest_proton : float
            Closest distance between the shower startpoint and proton points.
        """
        vertex_dist = -np.inf
        vertex = ia.vertex
        vertex_dist = np.linalg.norm(vertex - shower_p.start_point)
        return vertex_dist
    
    @staticmethod
    def convdist_relaxed(ia, shower_p, eps=0.6):
        """Helper function to compute the path-connected distance from the
        vertex to the closest shower point.

        Parameters
        ----------
        ia : RecoInteraction
            Reco interaction to apply the conversion distance cut.
        shower_p : RecoParticle
            Member particle of the interaction, assumed to be the primary
            electron/gamma shower.
        eps : float, default 0.6
            Maximum distance between two samples for one to be considered
            as in the neighborhood of the other (DBSCAN).

        Returns
        -------
        _type_
            _description_
        """
        pts = np.vstack([p.points for p in ia.particles])
        labels = np.hstack([np.ones(p.size) * p.id for p in ia.particles]).astype(int)
        model = DBSCAN(eps=eps, min_samples=1).fit(pts)
        clusts = model.labels_

        conversion_dist = np.inf
        
        shower_clusts = clusts[labels == shower_p.id]
        for c in np.unique(shower_clusts):
            # print(c, (clusts == c).sum())
            mask = (clusts == c)
            dists = np.linalg.norm(pts[mask] - ia.vertex, axis=1)
            sep = dists.min()
            conversion_dist = min(sep, conversion_dist)

        return conversion_dist
            
                
    # @staticmethod
    def compute_angular_criterion(self, p, vertex, eps, min_samples):
        """Compute the angular criterion for the given primary electron shower.

        Parameters
        ----------
        p : RecoParticle
            Primary electron shower to check for multi-arm.
        vertex : np.ndarray
            Vertex of the interaction with shape (3, )
        eps : float
            Maximum distance between two samples for one to be considered
            as in the neighborhood of the other (DBSCAN).
        min_samples : int
            The number of samples (or total weight) in a neighborhood 
            for a point to be considered as a core point (DBSCAN).

        Returns
        -------
        max_angle : float
            Maximum angle between the mean cluster direction vectors 
            of the shower points (degrees)
        """
        points = p.points
        depositions = p.depositions

        # Draw vector from startpoint to all 
        v = points - vertex
        v_norm = np.linalg.norm(v, axis=1)
        # If all vectors are zero, return 0
        if (v_norm > 0).sum() == 0:
            return 0
        # Normalize the vectors
        directions = v[v_norm > 0] / v_norm[v_norm > 0].reshape(-1, 1)
        
        # Filter out points that give zero vectors
        points = points[v_norm > 0]
        depositions = depositions[v_norm > 0]
        
        # If there are no valid directions, return -inf (will never be rejected)
        if directions.shape[0] < 1:
            return -np.inf
        
        # Run DBSCAN clustering on the unit sphere
        model = DBSCAN(eps=eps, 
                       min_samples=min_samples, 
                       metric='cosine').fit(directions)
        clusts, counts = np.unique(model.labels_, return_counts=True)
        
        if self.sort_by == 'energy':
            if not np.all(clusts >= 0): # If there are outliers
                labels = np.array(model.labels_ + 1, dtype=int)
            else:
                labels = np.array(model.labels_, dtype=int)
            energies = np.bincount(labels, weights=depositions)
            perm = np.argsort(energies)[::-1]
            clusts, counts = clusts[perm], counts[perm]
        elif self.sort_by == 'voxel_counts':
            perm = np.argsort(counts)[::-1]
            clusts, counts = clusts[perm], counts[perm]
        else:
            raise ValueError('Invalid sorting mode {}, must be either "energy" or "voxel_counts".'.format(self.sort_by))
        
        if self.largest_two:
            clusts, counts = clusts[:2], counts[:2]
        
        vecs = []
        for i, c in enumerate(clusts):
            # Skip noise points that have cluster label -1
            if c == -1: continue
            # Compute the mean direction vector of the cluster
            v = directions[model.labels_ == c].mean(axis=0)
            vecs.append(v / np.linalg.norm(v))
        if len(vecs) == 0:
            return -np.inf
        vecs = np.vstack(vecs)
        cos_dist = cosine_similarity(vecs)
        # max_angle ranges from 0 (parallel) to 2 (antiparallel)
        max_angle = np.clip((1.0 - cos_dist).max(), a_min=0, a_max=2)
        max_angle_deg = np.rad2deg(np.arccos(1 - max_angle))
        # counts = counts[1:]
        return max_angle_deg
    
    
class ShowerStartpointCorrectionProcessor(PostBase):
    """Correct the startpoint of the primary EM shower by 
    finding the closest point to the vertex.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'showerstart_correction_processor'

    # Alternative allowed names of the post-processor
    aliases = ('reco_shower_startpoint_correction',)
    
    def __init__(self, threshold=1.0):
        """Specify the EM shower conversion distance threshold and
        the type of vertex to use for the distance calculation.

        Parameters
        ----------
        threshold : float, default -1.0
            If EM shower has a conversion distance greater than this,
            its PID will be changed to PHOT_PID.
        """
        super().__init__('interaction', 'reco')
        self.threshold = threshold
        
    def process(self, data):
        """Update the shower startpoint using the closest point to the vertex.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for ia in data['reco_interactions']:
            for p in ia.particles:
                if (p.shape == SHOWR_SHP) and (p.is_primary):
                    new_point = self.correct_shower_startpoint(p, ia)
                    p.start_point = new_point
                    
                        
    @staticmethod                
    def correct_shower_startpoint(shower_p, ia):
        """Function to correct the shower startpoint by finding the closest
        point to a track.

        Parameters
        ----------
        shower_p : RecoParticle
            Primary EM shower to correct the startpoint.
        ia : RecoInteraction
            Reco interaction to use as the vertex estimate.

        Returns
        -------
        guess : np.ndarray
            (3, ) array of the corrected shower startpoint.
        """
        track_points = [p.points for p in ia.particles if p.shape == TRACK_SHP and p.is_primary]
        if track_points == []:
            return shower_p.start_point
        
        track_points = np.vstack(track_points)
        dist = cdist(shower_p.points.reshape(-1, 3), track_points.reshape(-1, 3))
        min_dist = dist.min()
        closest_idx, _ = np.where(dist == min_dist)
        if len(closest_idx) == 0:
            return shower_p.start_point
        guess = shower_p.points[closest_idx[0]]
        return guess
    
    
class MichelTaggingProcessor(PostBase):
    
    name = 'michel_tagging'
    aliases = ('michel_tagging_processor',)
    
    def __init__(self, r=15.0, segment_length=2.0, method='bin_pca', 
                 min_count=5, adj_threshold=1.0):
        """Post-processor to tag Michel electrons by checking the Bragg peak
        of the track adjacent to the shower startpoint.

        Parameters
        ----------
        r : float, default 15.0
            Distance of farthest track point to consider for bragg peak.
        segment_length : float, default 2.0
            Length of the track segment to consider for dedx calculation.
        method : str, default 'bin_pca'
            Method to use for dedx calculation.
        min_count : int, default 5
            Minimum number of points to consider for dedx calculation.
        """
        super().__init__('interaction', 'reco')
        self.segment_length = segment_length
        self.method = method
        self.min_count = min_count
        self.r = r
        self.adj_threshold = adj_threshold
        
    def process(self, data):
        """Compute the correlation of residual range vs. local dedx to use
        as a feature for Michel electron tagging. 

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for ia in data['reco_interactions']:
            
            for p in ia.particles:
                
                if p.pid == ELEC_PID and (p.shape == 0 or p.shape == 2):
                    y, x = self.check_bragg_peak(p, ia, r=self.r, 
                                                 adj_threshold=self.adj_threshold)
                    if (len(x) > 2) and (len(x) == len(y)):
                        bragg = pearsonr(x, y)[0]
                        p.adjacent_bragg_pearsonr = bragg
                    else:
                        p.adjacent_bragg_pearsonr = np.inf
                
            
    def check_bragg_peak(self, shower_p, nu_reco, r, adj_threshold=1.0):
        """Given a primary shower and a reco interaction, finds track points 
        near the shower startpoint within radius r and computes the 
        residual range and local dedx arrays for the track points. 
        
        The purpose of this module is to locate the Bragg peak of the track
        that is adjacent to the shower startpoint.

        Parameters
        ----------
        shower_p : RecoParticle
            Primary shower to check for Bragg peak.
        nu_reco : RecoInteraction
            RecoInteraction that contains the shower.
        r : float
            Radius to search for track points near the shower startpoint.

        Returns
        -------
        dedxs : np.ndarray
            Array of dedx values for the track points near the shower startpoint.
        rrs : np.ndarray
            Array of residual ranges for the track points near the shower startpoint.
        """

        track_points = []
        track_deps = []
        for p in nu_reco.particles:
            if p.shape == 1:
                track_points.append(p.points)
                track_deps.append(p.depositions)

        if len(track_points) > 0:
            track_points = np.vstack(track_points)
            track_deps = np.hstack(track_deps)
        else:
            return np.array([]), np.array([])
        
        dists = np.linalg.norm(track_points - shower_p.start_point, axis=1)
        if dists.min() > adj_threshold:
            return np.array([]), np.array([])
        
        mask = dists < r
        near_pts = track_points[mask]
        near_dps = track_deps[mask]
        
        if len(near_pts) == 0:
            return np.array([]), np.array([])

        adj_pt = track_points[np.argmin(dists)]

        dedxs, _, rrs, _, _, _ = get_track_segment_dedxs(near_pts, 
            near_dps, 
            adj_pt, 
            segment_length=self.segment_length, 
            method=self.method, 
            min_count=self.min_count)

        return dedxs, rrs