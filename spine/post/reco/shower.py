import numpy as np

from spine.utils.globals import (PHOT_PID, PROT_PID, PION_PID, ELEC_PID, 
                                 SHOWR_SHP, TRACK_SHP)

from spine.post.base import PostBase
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

<<<<<<< HEAD
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

=======
>>>>>>> develop
from spine.utils.gnn.cluster import cluster_dedx

__all__ = ['ConversionDistanceProcessor', 'ShowerMultiArmCheck', 
           'ShowerStartpointCorrectionProcessor', 'ShowerdEdXProcessor',
           'ShowerSpreadProcessor']


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
    
<<<<<<< HEAD
    def __init__(self, threshold=-1.0, vertex_mode='vertex_points', modify_inplace=True):
=======
    def __init__(self, threshold=-1.0, vertex_mode='vertex_points', inplace=True):
>>>>>>> develop
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
<<<<<<< HEAD
        self.modify_inplace = modify_inplace
=======
        self.inplace = inplace
>>>>>>> develop
        
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
            
            leading_shower, energy = None, -np.inf
            
            for p in ia.particles:
                
                if (p.shape == SHOWR_SHP and p.is_primary):
                    
                    if self.vertex_mode == 'protons':
                        criterion = self.convdist_protons(ia, p)
                    elif self.vertex_mode == 'vertex_points':
                        criterion = self.convdist_vertex_points(ia, p)
                    elif self.vertex_mode == 'vertex_startpoint':
                        criterion = self.convdist_vertex_startpoint(ia, p)
                    else:
                        raise ValueError('Invalid point mode')
                    
                    p.vertex_distance = criterion
                    p.shower_dedx = cluster_dedx(p.points,
                                              p.depositions,
                                              p.start_point,
                                              max_dist=3.0)
                    
                    if p.ke > energy:
                        leading_shower = p
                        energy = p.ke
                    
                    if p.pid == ELEC_PID:
                        
                        if self.modify_inplace:
                            if criterion >= self.threshold:
                                p.pid = PHOT_PID
                            
            if leading_shower is None:
                ia.vertex_distance = -np.inf
                ia.leading_shower_dedx = -np.inf
                ia.leading_shower_num_fragments = -1
            else:
                ia.vertex_distance = leading_shower.vertex_distance
                ia.leading_shower_dedx = cluster_dedx(leading_shower.points,
                                              leading_shower.depositions,
                                              leading_shower.start_point,
                                              max_dist=3.0)
                ia.leading_shower_num_fragments = leading_shower.num_fragments
            
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

    # Name of the post-processor (as specified in the configuration)
    name = 'shower_multi_arm_check'

    # Alternative allowed names of the post-processor
    aliases = ('shower_multi_arm',)
    
    def __init__(self, threshold=70, min_samples=20, eps=0.02, inplace=True):
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
        self.inplace = inplace
        
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
            
            leading_shower, energy = None, -np.inf
            
            for p in ia.particles:
                
                if p.shape == SHOWR_SHP and p.is_primary:
                    
                    angle = self.compute_angular_criterion(p, ia.vertex, 
                                                     eps=self.eps, 
                                                     min_samples=self.min_samples)
                    
                    _, _, spread = shower_quality_check(p, ia.vertex)
                    p.shower_spread = spread
                    
                    p.shower_split_angle = angle
                    if self.inplace:
                        if angle > self.threshold:
                            p.pid = PHOT_PID
                
                if p.pid == ELEC_PID and p.is_primary and (p.shape == SHOWR_SHP):

                    if self.modify_inplace:
                        if angle > self.threshold:
                            p.pid = PHOT_PID
            
            if leading_shower is None:
                ia.shower_split_angle = -np.inf
                ia.shower_spread = -np.inf
            else:
                ia.shower_split_angle = leading_shower.shower_split_angle
                ia.shower_spread = leading_shower.shower_spread
            
                
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


class ShowerdEdXProcessor(PostBase):
    """Compute the dEdX of the primary EM shower
    by summing the energy depositions along the shower trunk and dividing
    by the total length of the trunk.
    """
    
    name = 'shower_dedx_processor'
    aliases = ('shower_dedx',)
    
    def __init__(self, threshold=4.0, max_dist=3.0, inplace=True):
        """Specify the EM shower dEdX threshold.

        Parameters
        ----------
        threshold : float, default 4.0
            If the dEdX of the shower is greater than this, the shower
            will be considered a photon.
        inplace : bool, default True
            If True, the processor will update the reco interaction in-place.
        """
        super().__init__('interaction', 'reco')
        self.threshold = threshold
        self.max_dist = max_dist
        self.inplace = inplace
        
    def process(self, data):
        """Compute the shower dEdX and modify the PID if inplace=True.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for ia in data['reco_interactions']:
            
            leading_shower, max_ke = None, -np.inf
            
            for p in ia.particles:
                if (p.shape == SHOWR_SHP) and (p.is_primary):
                    dedx = cluster_dedx(p.points, 
                                        p.depositions, 
                                        p.start_point, 
                                        max_dist=self.max_dist)
                    p.shower_dedx = dedx
                    
                    if p.ke > max_ke:
                        leading_shower = p
                        max_ke = p.ke
                    
                    if self.inplace:
                        if dedx >= self.threshold:
                            p.pid = PHOT_PID
            
            if leading_shower is not None:
                ia.leading_shower_dedx = leading_shower.shower_dedx
            else:
                ia.leading_shower_dedx = -1.
            
            
class ShowerSpreadProcessor(PostBase):
    """Compute the spread of the primary EM shower
    by computing the RMS of the energy depositions along the shower trunk.
    """
    
    name = 'shower_spread_processor'
    aliases = ('shower_spread',)
    
    def __init__(self, threshold=0.043, length_scale=14.0, inplace=True):
        """Specify the EM shower spread threshold.

        Parameters
        ----------
        threshold : float, default 4.0
            If the spread of the shower is greater than this, the shower
            will be considered a photon.
        inplace : bool, default True
            If True, the processor will update the reco interaction in-place.
        """
        super().__init__('interaction', 'reco')
        self.threshold = threshold
        self.length_scale = length_scale
        self.inplace = inplace
        
    def process(self, data):
        """Compute the shower spread and modify the PID if inplace=True.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for ia in data['reco_interactions']:
            
            leading_shower, max_ke = None, -np.inf
            
            for p in ia.particles:
                if (p.shape == SHOWR_SHP) and (p.is_primary):
                    spread = compute_shower_spread(p.points,
                                                   ia.vertex,
                                                   l=self.length_scale)
                    p.shower_spread = spread
                    
                    if p.ke > max_ke:
                        leading_shower = p
                        max_ke = p.ke
                    
                    if self.inplace:
                        if spread >= self.threshold:
                            p.pid = PHOT_PID
            
            if leading_shower is not None:
                ia.leading_shower_spread = leading_shower.shower_spread
            else:
                ia.leading_shower_spread = -1.
                
                

def compute_shower_spread(points, vertex, l=14.0):
    """Compute the spread of the shower by computing the mean direction and
    the weighted average cosine distance with respect to the mean direction.
    
    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of the shower points.
    l : float, default 14.0
        Length scale for the exponential weighting.
        
    Returns
    -------

    spread : float
        Spread cut parameter of the shower.
    """
    dists = np.linalg.norm(points - vertex, axis=1)
    # Compute Spread (whole shower)
    directions = (points - vertex) / dists.reshape(-1, 1)
    weights = np.exp(- dists / l)
    mean_direction = np.average(directions, weights=weights, axis=0)
    if np.linalg.norm(mean_direction) > 1e-6:
        mean_direction /= np.linalg.norm(mean_direction)

        cosine = 1 - np.sum(directions * mean_direction.reshape(1, -1), axis=1)
        spread = np.average(cosine, weights=weights)
    else:
        spread = -1. 

    return spread
