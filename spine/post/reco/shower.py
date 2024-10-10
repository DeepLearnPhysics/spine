import numpy as np

from spine.utils.globals import (PHOT_PID, PROT_PID, PION_PID, ELEC_PID, 
                                 SHOWR_SHP, TRACK_SHP)

from spine.post.base import PostBase
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ['ConversionDistanceProcessor', 'ShowerMultiArmCheck', 
           'ShowerStartpointCorrectionProcessor']


class ConversionDistanceProcessor(PostBase):
    """Enforce additional constraint on valid primary electron showers
    using vertex-to-shower separation distance. 
    
    NOTE: This processor can only change reco electron shower pid to
    photon pid depending on the distance threshold. 
    """
    name = 'shower_conversion_distance'
    aliases = ['shower_separation_processor']
    
    def __init__(self, threshold=-1.0, vertex_mode='vertex'):
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
            for p in ia.particles:
                if (p.shape == 0 and p.pid == 1 and p.is_primary):
                    if self.vertex_mode == 'protons':
                        criterion = self.convdist_protons(ia, p)
                    elif self.vertex_mode == 'vertex_points':
                        criterion = self.convdist_vertex_points(ia, p)
                    elif self.vertex_mode == 'vertex_startpoint':
                        criterion = self.convdist_vertex_startpoint(ia, p)
                    else:
                        raise ValueError('Invalid point mode')
                    p.vertex_distance = criterion
                    if criterion >= self.threshold:
                        p.pid = PHOT_PID
            
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
    def convdist_vertex_points(ia, shower_p):
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
        vertex = ia.vertex
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
    
    
class ShowerMultiArmCheck(PostBase):
    """Check whether given primary electron candidate is likely
    to be a merged multi-particle shower.
    
    This processor computes direction vectors of the shower points
    from the shower start and performs DBSCAN clustering on the unit sphere
    using the cosine distance metric. If there are more than one cluster that
    has a mean direction vector outside a certain angular threshold, the
    shower is considered to be a multi-arm shower and is rejected as 
    a valid primary electron candidate.
    
    NOTE: This processor can only change reco electron shower pid to
    photon pid depending on the angle threshold. 
    """
    name = 'shower_multi_arm_check'
    aliases = ['shower_multi_arm']
    
    def __init__(self, threshold=70, min_samples=20, eps=0.02):
        """Specify the threshold for the number of arms of showers.

        Parameters
        ----------
        threshold : float, default 70 (deg)
            If the electron shower's leading and subleading angle are
            separated by more than this, the shower is considered to be
            invalid and its PID will be changed to PHOT_PID.
        min_samples : int, default 20
            The number of samples (or total weight) in a neighborhood 
            for a point to be considered as a core point (DBSCAN).
        eps : float, default 0.02
            Maximum distance between two samples for one to be considered
            as in the neighborhood of the other (DBSCAN).
        """
        super().__init__('interaction', 'reco')
        
        self.threshold = threshold
        self.min_samples = min_samples
        self.eps = eps
        
    def process(self, data):
        """Update reco interaction topologies using the shower multi-arm check.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for ia in data['reco_interactions']:
            # Loop over particles, select the ones that pass a threshold
            for p in ia.particles:
                if p.pid == ELEC_PID and p.is_primary and (p.shape == 0):
                    angle = self.compute_angular_criterion(p, ia.vertex, 
                                                     eps=self.eps, 
                                                     min_samples=self.min_samples)
                    p.shower_split_angle = angle
                    if angle > self.threshold:
                        p.pid = PHOT_PID
                
    @staticmethod
    def compute_angular_criterion(p, vertex, eps, min_samples):
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
        
        # If there are no valid directions, return 0
        if directions.shape[0] < 1:
            return 0
        
        # Run DBSCAN clustering on the unit sphere
        model = DBSCAN(eps=eps, 
                       min_samples=min_samples, 
                       metric='cosine').fit(directions)
        clusts, counts = np.unique(model.labels_, return_counts=True)
        perm = np.argsort(counts)[::-1]
        clusts, counts = clusts[perm], counts[perm]
        
        vecs = []
        for i, c in enumerate(clusts):
            # Skip noise points that have cluster label -1
            if c == -1: continue
            # Compute the mean direction vector of the cluster
            v = directions[model.labels_ == c].mean(axis=0)
            vecs.append(v / np.linalg.norm(v))
        if len(vecs) == 0:
            return 0
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
    name = 'showerstart_correction_processor'
    aliases = ['reco_shower_startpoint_correction']
    
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
            vertex = ia.vertex
            for p in ia.particles:
                if (p.shape == SHOWR_SHP) and (p.is_primary):
                    new_point = self.correct_shower_startpoint(p, ia)
                    p.start_point = new_point
                    
                        
    @staticmethod                
    def correct_shower_startpoint(shower_p, ia):
        """Function to correct the shower startpoint by finding the closest
        point to the vertex.

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
