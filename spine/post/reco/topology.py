"""Particle topology and geometry based reco modules."""

import numpy as np
from spine.post.base import PostBase
from spine.utils.gnn.cluster import (cluster_dedx_dir, 
                                     cluster_dedx_DBScan_PCA, 
                                     cluster_dedx)

from sklearn.decomposition import PCA
from scipy.stats import pearsonr

__all__ = ['ParticleMultiArmCheck', 
           'ParticledEdXProcessor', 
           'ParticleSpreadProcessor',
           'ParticleTrunkStraightnessProcessor']


class ParticleMultiArmCheck(PostBase):
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
    name = 'particle_multi_arm_check'

    # Alternative allowed names of the post-processor
    aliases = ('particle_multi_arm',)
    
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
            
            for p in ia.particles:
                if p.is_primary:
                    angle = self.compute_angular_criterion(p, ia.vertex, 
                                                        eps=self.eps, 
                                                        min_samples=self.min_samples)
                    
                    p.split_angle = angle


class ParticledEdXProcessor(PostBase):
    """Compute the dEdX of the primary EM shower
    by summing the energy depositions along the shower trunk and dividing
    by the total length of the trunk.
    """
    
    name = 'particle_dedx_processor'
    aliases = ('start_dedx',)
    
    def __init__(self, threshold=4.0, max_dist=3.0, mode='direction'):
        """Specify the EM shower dEdX threshold.

        Parameters
        ----------
        threshold : float, default 4.0
            If the dEdX of the shower is greater than this, the shower
            will be considered a photon.
        max_dist : float, default 3.0
            Maximum distance to consider for the dEdX calculation.
        mode : str, default 'direction'
            Method to use for dEdX calculation.
        inplace : bool, default True
            If True, the processor will update the reco interaction in-place.
        """
        super().__init__('interaction', 'reco')
        self.threshold = threshold
        self.max_dist = max_dist
        self.mode = mode
        
    def process(self, data):
        """Compute the shower dEdX and modify the PID if inplace=True.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for ia in data['reco_interactions']:
            
            for p in ia.particles:
                    
                if p.is_primary:
                    
                    if self.mode == 'default':
                        dedx = cluster_dedx(p.points, p.depositions, p.start_point, max_dist=self.max_dist)
                    elif self.mode == 'dbscan':
                        dedx = cluster_dedx_DBScan_PCA(p.points, p.depositions, p.start_point, 
                                                    self.max_dist, simple=True)
                    elif self.mode == 'direction':
                        dedx = cluster_dedx_dir(p.points,
                                                p.depositions,
                                                p.start_point,
                                                p.start_dir,
                                                dedx_dist=self.max_dist,
                                                simple=True)
                    else:
                        raise ValueError('Invalid dEdX calculation mode')
                    p.start_dedx = dedx
            
            
class ParticleSpreadProcessor(PostBase):
    """Compute the directional spread of a particle by computing 
    the mean direction and the weighted average cosine distance
    with respect to the mean direction.
    
    Also computes the axial spread, which is the pearson R correlation
    coefficient between the distance of the shower points from the startpoint
    along the shower axis and the perpendicular distance from the shower axis.
    """
    
    name = 'particle_spread_processor'
    aliases = ('particle_spread',)
    
    def __init__(self, threshold=0.043, length_scale=14.0, 
                 refvox_mode='vertex'):
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
        self.refvox_mode = refvox_mode
        
    def process(self, data):
        """Compute the shower spread and modify the PID if inplace=True.

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the reco interactions
        for ia in data['reco_interactions']:
            
            for p in ia.particles:
                
                if p.is_primary:
                    
                    if self.refvox_mode == 'vertex':
                    
                        p.directional_spread = compute_particle_spread(p.points,
                                                                    ia.vertex,
                                                                    l=self.length_scale)
                        p.axial_spread = compute_axial_pearsonr(p, ia.vertex)
                    
                    elif self.refvox_mode == 'startpoint':
                        
                        p.directional_spread = compute_particle_spread(p.points,
                                                                    p.start_point,
                                                                    l=self.length_scale)
                        p.axial_spread = compute_axial_pearsonr(p, p.start_point)
                        
                    else:
                        raise ValueError('Invalid reference voxel mode')
    
    
def compute_axial_pearsonr(shower_p, refpoint):
    """Compute the pearson R correlation coefficient between the
    distance of the shower points from the startpoint along the shower
    axis and the perpendicular distance from the shower axis.

    Parameters
    ----------
    shower_p : RecoParticle
        Primary EM shower to compute the axial pearson R.
    refpoint : np.ndarray
        Reference point to compute the axial pearson R.

    Returns
    -------
    The pearson R correlation coefficient between (-1, 1)
    """
    
    if len(shower_p.points) < 3:
        return -np.inf
    
    startpoint = refpoint
    v0 = shower_p.start_dir
    
    dists = np.linalg.norm(shower_p.points - startpoint, axis=1)
    v = (startpoint - shower_p.points) - np.sum((startpoint - shower_p.points) * v0, axis=1, keepdims=True) \
      * np.broadcast_to(v0, shower_p.points.shape)
    perps = np.linalg.norm(v, axis=1)
    
    out = pearsonr(dists, perps)
    return out[0]


def compute_particle_spread(points, vertex, l=14.0):
    """Compute the spread of the particle by computing the mean direction and
    the weighted average cosine distance with respect to the mean direction.
    
    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of the particle points.
    l : float, default 14.0
        Length scale for the exponential weighting.
        
    Returns
    -------

    spread : float
        Spread cut parameter of the particle.
    """
    dists = np.linalg.norm(points - vertex, axis=1)
    mask = dists > 0
    if mask.sum() == 0:
        return -np.inf

    # Compute Spread (whole shower)
    directions = (points[mask] - vertex) / dists[mask].reshape(-1, 1)
    weights = np.clip(np.exp(- dists[mask] / l), min=1e-6)
    mean_direction = np.average(directions, weights=weights, axis=0)
    mean_direction /= np.linalg.norm(mean_direction)
    cosine = 1 - np.sum(directions * mean_direction.reshape(1, -1), axis=1)
    spread = np.average(cosine, weights=weights)

    return spread


class ParticleTrunkStraightnessProcessor(PostBase):
    """Compute the validity of the shower trunk by computing the PCA
    principal explained variance ratio of the shower points.
    """
    
    name = 'particle_trunk_strightness'
    aliases = ('particle_trunk_processor',)
    
    def __init__(self, threshold=0.0, r=3.0, n_components=3, inplace=False):
        """Specify the EM shower spread thresholds.

        Parameters
        ----------
        threshold : float
            If the spread of the shower trunk is less than this value, the
            particle is labeled as invalid.
        inplace : bool, default True
            If True, the processor will update the reco interaction in-place.
        """
        super().__init__('interaction', 'reco')
        self.threshold = threshold
        self.r = r
        self.n_components = n_components
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
            
            for p in ia.particles:
                
                if p.is_primary:
                    
                    p.trunk_straightness = compute_trunk_straightness(p,
                                                                      r=self.r,
                                                                      n_components=self.n_components)
                        
                    if self.inplace:
                        if p.trunk_straightness < self.threshold:
                            p.is_valid = False
                            p.is_primary = False
                    
                
def compute_trunk_straightness(shower_p, r=3.0, n_components=3):
    """Helper function to compute the validity of the shower trunk
    by computing the PCA principal explained variance ratio. 

    Parameters
    ----------
    shower_p : RecoParticle
        Primary EM shower to compute the trunk validity. (Doesn't need
        to be a shower, can be any particle with points)
    r : float, default 3.0
        Radius to search for points near the startpoint.
    n_components : int, default 3
        Number of components to keep

    Returns
    -------
    The first explained variance ratio of the PCA of the shower trunk.
    """
    dists = np.linalg.norm(shower_p.points - shower_p.start_point, axis=1)
    mask = dists < r
    pts = shower_p.points[mask]
    if len(pts) <= n_components:
        return -np.inf
    else:
        pca = PCA(n_components=n_components)
        pca.fit(pts)
        return pca.explained_variance_ratio_[0]