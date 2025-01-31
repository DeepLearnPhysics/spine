'''Module to evaluate diagnostic metrics on showers.'''

import numpy as np

from spine.ana.base import AnaBase
from spine.utils.gnn.cluster import cluster_dedx2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

__all__ = ['ShowerStartSingleParticle']

def cluster_dedx2_with_PCA(voxels,
                 values,
                 start,
                 dedx_dist=3, cont_dist=5, detailed=True, num_intervals=10):
    # If max_dist is set, limit the set of voxels to those within a sphere of radius max_dist                                
    assert voxels.shape[1] == 3, (
            "The shape of the input is not compatible with voxel coordinates.")

    # If start point is not in voxels, assign the closest point within voxels
    # as the startpoint
    if start not in voxels:
        dists = np.linalg.norm(voxels - start, axis=1)
        perm = np.argsort(dists)
        start = voxels[perm[0]]

    # distance from the startpoint
    dist_mat = cdist(start.reshape(1,-1), voxels).flatten()

    # legacy dedx
    #if dedx_dist > 0:
    voxels_dedx = voxels[dist_mat <= dedx_dist]
    #print("thr: ", max_dist, ", num vox: ", len(voxels))
    if len(voxels_dedx) < 2:
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    values_dedx = values[dist_mat <= dedx_dist]
    dist_dedx = dist_mat[dist_mat <= dedx_dist]
    # Calculate sum of values                                                                                                                                                                                                         
    sum_dedx = np.sum(values_dedx)
    # Calculate max distance for dedx
    max_dist_dedx = np.max(dist_dedx)

    
    # continuity check 
    voxels_cont = voxels[dist_mat <= cont_dist]
    values_cont = values[dist_mat <= cont_dist]
    dist_cont = dist_mat[dist_mat <= cont_dist]
    # Perform DBSCAN clustering
    # parameters are not yet tuned
    eps = 0.7
    min_samples = 5
    dbscan = DBSCAN(eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(voxels_cont)
    #clusts, counts = np.unique(cluster_labels, return_counts=True)
    num_clust = max(0, max(cluster_labels)+1)

    # fining the dbscan cluster containing the startpoint
    start_clust = -1
    true_p_clust1 = -1
    true_p_clust2 = -1
    true_p_clust3 = -1
    for i in range(num_clust):
        if start in voxels_cont[cluster_labels==i]:
            start_clust = i
        if i==0:
            true_p_clust1 = len(voxels_cont[cluster_labels==i])
        if i==1:
            true_p_clust2 = len(voxels_cont[cluster_labels==i])
        if i==2:
            true_p_clust3 = len(voxels_cont[cluster_labels==i])
            
    voxels_clust = voxels_cont[cluster_labels==start_clust]
    values_clust = values_cont[cluster_labels==start_clust]
    dist_clust = dist_cont[cluster_labels==start_clust]
    voxels_clust = voxels_clust[dist_clust<dedx_dist]
    values_clust = values_clust[dist_clust<dedx_dist]
    dist_clust = dist_clust[dist_clust<dedx_dist]
    

    if len(voxels_clust)<3:
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(voxels_clust)
    p_axis = pca.components_[0]
    p_fit = pca.explained_variance_ratio_[0]
    
    # Project voxels onto the principal axis
    p_voxels = np.dot(voxels_clust - np.mean(voxels_clust, axis=0), p_axis)

    min_proj = np.min(p_voxels)
    max_proj = np.max(p_voxels)
    mask = (p_voxels >= min_proj) & (p_voxels <= max_proj)
    
    #print("len: ", len(values_dedx), "proj len: ", len(values_dedx[mask]))
    p_sum = np.sum(values_clust[mask])
    p_length = max_proj - min_proj

    return sum_dedx, max_dist_dedx, p_sum, p_length, p_fit, num_clust, start_clust, true_p_clust1, true_p_clust2, true_p_clust3

class ShowerStartSingleParticle(AnaBase):
    """This analysis script computes the dE/dx value within some distance
    from the start point of an EM shower object.

    This is a useful diagnostic tool to evaluate the calorimetric separability
    of different EM shower types (electron vs photon), which are expected to
    have different dE/dx patterns near their start point.
    """

    # Name of the analysis script (as specified in the configuration)
    name = 'shower_start_singlep'

    

    def __init__(self, radius, is_photon=False, **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        radius : Union[float, List[float]]
            Radius around the start point for which evaluate dE/dx
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__('particle', 'both', **kwargs)

        # Store parameters
        self.radius = radius

        self.is_photon = is_photon

        # Initialize the CSV writer(s) you want
        for obj in self.obj_type:
            self.initialize_writer(obj)
        self.update_keys({'clust_label_adapt': True, 'meta': True, 'particles': True})
        self.units = 'cm'

    def process(self, data):
        """Evaluate shower start dE/dx for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the keys you want
        
        if (len(data['truth_particles']) > 0):
            
            out_dict = {}
            out_dict['index'] = data['index']
            #print(data['index'])
            out_dict['file_index'] = data['file_index']
            out_dict['file_entry_index'] = data['file_entry_index']

            out_dict['num_true'] = len(data['truth_particles'])
            # True Showers

            out_dict['true_crea1'] = None
            out_dict['true_crea2'] = None
            out_dict['true_crea3'] = None

            out_dict['true_trackid1'] = None
            out_dict['true_trackid2'] = None
            out_dict['true_trackid3'] = None

            true_shower = data['truth_particles'][0]
            out_dict['true_creation_process'] = -1

            out_dict['is_pos'] = -1
            # photon cleanup
            # if primary's daughter is created via compton scattering, discard
            if self.is_photon:
                is_positron = False
                true_particles = data['particles']
                for part in true_particles:
                    #print("pdg: ", part.parent_pdg_code)
                    if part.parent_pdg_code == -11:
                        is_positron = True
                out_dict['is_pos'] = True if is_positron else False

                for i in range(min(len(data['truth_particles']), 3)):
                    if i==0:
                        out_dict['true_crea1'] = data['truth_particles'][i].creation_process
                        out_dict['true_trackid1'] = data['truth_particles'][i].pid
                    elif i==1:
                        out_dict['true_crea2'] = data['truth_particles'][i].creation_process
                        out_dict['true_trackid2'] = data['truth_particles'][i].pid
                    elif i==2:
                        out_dict['true_crea3'] = data['truth_particles'][i].creation_process
                        out_dict['true_trackid3'] = data['truth_particles'][i].pid
                #    return

                out_dict['true_creation_process'] = true_shower.creation_process
                #    return
#                group_id = data['truth_particles'][0].group_id

 #               true_particles = data['particles']
 #               for part in true_particles:
 #                   if part.group_id == group_id:
#                       out_dict['true_group_creation']=out_dict['true_group_creation']+part.creation_process
                
            startpoint = true_shower.start_point
            out_dict['true_particle_id'] = true_shower.id
            out_dict['true_particle_pdg'] = true_shower.pdg_code
            out_dict['true_particle_energy_init'] = true_shower.energy_init
            out_dict['true_particle_energy_deposit'] = true_shower.energy_deposit
            out_dict['true_is_contained'] = true_shower.is_contained
            
            t_de_1, t_dx_1, p_de, p_dx, p_fit, num_clust, start_clust, true_p_clusts0, true_p_clusts1, true_p_clusts2  = cluster_dedx2_with_PCA(true_shower.points, 
                                  true_shower.depositions, 
                                  startpoint, 
                                  dedx_dist=self.radius)
            
                        
            out_dict['true_de_1'] = t_de_1
            out_dict['true_dx_1'] = t_dx_1
            out_dict['true_PCA_de'] = p_de
            out_dict['true_PCA_dx'] = p_dx
            out_dict['true_p_fit'] = p_fit
            out_dict['true_p_num_clust'] = num_clust
            #out_dict['true_l_1'] = t_l_1
            out_dict['true_match_overlap'] = -1
            out_dict['true_p_start_clust'] = start_clust
            out_dict['true_p_clust_size1'] = true_p_clusts0
            out_dict['true_p_clust_size2'] = true_p_clusts1
            out_dict['true_p_clust_size3'] = true_p_clusts2
            
            match_id = -1
            if true_shower.match_overlaps is not None and len(true_shower.match_overlaps)>0:
                out_dict['true_match_overlap'] = true_shower.match_overlaps[0]
                match_id = true_shower.match_ids[0]
            # Reco Showers, reco points
            
            if len(data['reco_particles']) == 0:
                return
            if match_id == -1:
                return

            reco_shower = data['reco_particles'][match_id]
            startpoint = reco_shower.start_point
            out_dict['reco_particle_id'] = reco_shower.id
            # out_dict['reco_particle_energy_init'] = reco_shower.energy_init
            out_dict['reco_particle_energy_deposit'] = reco_shower.calo_ke
            out_dict['reco_is_contained'] = reco_shower.is_contained
            out_dict['reco_match_overlap'] = -1
            if reco_shower.match_overlaps is not None and len(reco_shower.match_overlaps)>0:
                out_dict['reco_match_overlap'] = reco_shower.match_overlaps[0]

            #print(data.keys())    
            reco_points = data['clust_label_adapt'][:,1:4]
            #print(reco_points)
            reco_vals = data['clust_label_adapt'][:,4]
            pos = reco_vals>0.

            reco_vals = reco_vals[pos]
            reco_points = reco_points[pos]

            #print(len(reco_vals))
            #for i in range(len(reco_vals)):
            #    if reco_vals[i]<0.:
            #    
            reco_points=data['meta'].to_cm(reco_points)
                        
            de_0, dx_0, l_0 = cluster_dedx2(reco_shower.points, 
                                  reco_shower.depositions,
                                     startpoint, 
                                     max_dist=self.radius)
            out_dict['reco_de_0'] = de_0
            out_dict['reco_dx_0'] = dx_0
            out_dict['reco_l_0'] = l_0
            
            de_1, dx_1, l_1 = cluster_dedx2(reco_points,
                                  reco_vals,
                                  startpoint,
                                     max_dist=self.radius)
            out_dict['reco_de_1'] = de_1
            out_dict['reco_dx_1'] = dx_1
            out_dict['reco_l_1'] = l_1
            
            # dedx with true startpoint
            de_02, dx_02, l_02 = cluster_dedx2(reco_shower.points, 
                                  reco_shower.depositions, 
                                  true_shower.start_point, 
                                     max_dist=self.radius)
            out_dict['reco_de_02'] = de_02
            out_dict['reco_dx_02'] = dx_02
            out_dict['reco_l_02'] = l_02
            
            de_2, dx_2, l_2 = cluster_dedx2(reco_points,
                                  reco_vals,
                                  true_shower.start_point,
                                     max_dist=self.radius)
            out_dict['reco_de_2'] = de_2
            out_dict['reco_dx_2'] = dx_2
            out_dict['reco_l_2'] = l_2
            # dedx with point closest to true startpoint
            
            dists = np.linalg.norm(reco_shower.points - true_shower.start_point, axis=1)
            perm = np.argsort(dists)
            closest_point = reco_shower.points[perm[0]]

            de_03, dx_03, l_03  = cluster_dedx2(reco_shower.points,                                                                                                                                
                       reco_shower.depositions,                                                                                                                                
                      closest_point,                                                                                                                                          
                       max_dist=self.radius)
            out_dict['reco_de_03'] = de_03
            out_dict['reco_dx_03'] = dx_03
            out_dict['reco_l_03'] = l_03
            
            de_3, dx_3, l_3 = 0., 0., 0.
            if len(reco_points)>0.:
                dists = np.linalg.norm(reco_points - true_shower.start_point, axis=1)
                perm = np.argsort(dists)
                closest_point = reco_points[perm[0]]

                #de_3, dx_3 = cluster_dedx2(reco_shower.points,
                #                      reco_shower.depositions,
                #                      closest_point,
                #                        max_dist=self.radius)
                de_3, dx_3, l_3 = cluster_dedx2(reco_points,
                                  reco_vals,
                                  closest_point,
                                    max_dist=self.radius)


            out_dict['reco_de_3'] = de_3
            out_dict['reco_dx_3'] = dx_3
            out_dict['reco_l_3'] = l_3
            
            self.append('particle', **out_dict)
