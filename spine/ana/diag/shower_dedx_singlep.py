'''Module to evaluate diagnostic metrics on showers.'''

import numpy as np

from spine.ana.base import AnaBase
from spine.utils.gnn.cluster import cluster_dedx_legacy, cluster_dedx_DBScan_PCA, cluster_dedx_dir
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

__all__ = ['ShowerStartSingleParticle']

def cluster_dedx2_with_PCA_temp(voxels,
                 values,
                 start,
                 dedx_dist=3, cont_dist=5, detailed=True, num_intervals=10):
    # If max_dist is set, limit the set of voxels to those within a sphere of radius max_dist                                
    assert voxels.shape[1] == 3, (
            "The shape of the input is not compatible with voxel coordinates.")

    # If start point is not in voxels, assign the closest point within voxels
    # as the startpoint
    if not np.isclose(start, voxels, atol=1e-2).all(axis=1).any():
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
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., [0., 0., 0.]
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
    eps = 0.59
    min_samples = 1
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
        if np.isclose(start, voxels_cont[cluster_labels==i], atol=1e-2).all(axis=1).any():
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
    
    voxels_clust = voxels_clust[dist_clust<=dedx_dist]
    values_clust = values_clust[dist_clust<=dedx_dist]
    dist_clust = dist_clust[dist_clust<=dedx_dist]

    if len(voxels_clust)<3:
        return sum_dedx, max_dist_dedx, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., [0., 0., 0.]
    dedx_clust_dist = np.max(dist_clust)
    # include dE from other clusters
    voxels_inc = voxels_cont[dist_cont<=dedx_clust_dist]
    values_inc = values_cont[dist_cont<=dedx_clust_dist]
    dist_inc = dist_cont[dist_cont<=dedx_clust_dist]

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

    # inclusive voxels
    p_voxels_inc = np.dot(voxels_inc - np.mean(voxels_clust, axis=0), p_axis)
    min_proj_inc = np.min(p_voxels_inc)
    max_proj_inc = np.max(p_voxels_inc)
    #mask_inc = (p_voxels_inc >= min_proj_inc) & (p_voxels_inc <= max_proj_inc)
    mask_inc = (p_voxels_inc >= min_proj) & (p_voxels_inc <= max_proj)
    #print("len: ", len(values_dedx), "proj len: ", len(values_dedx[mask]))
    p_sum = np.sum(values_clust[mask])

    voxels_de = voxels_inc[mask_inc]
    values_de = values_inc[mask_inc]
    p_sum_inc = np.sum(values_de)
    p_length = max_proj - min_proj
    p_length_inc = max_proj_inc - min_proj_inc

    voxels_sp = voxels_de - np.mean(voxels_clust, axis=0)
    p_voxels_sp = np.dot(voxels_sp, p_axis)
    vectors_to_axis = voxels_sp - np.outer(p_voxels_sp, p_axis)
    spread = np.linalg.norm(vectors_to_axis, axis=1)
    spread = sum(spread)/len(voxels_sp)
    #print("spread: ", spread)
    
    return sum_dedx, max_dist_dedx, p_sum, p_length, p_sum_inc, p_length_inc, spread, p_fit, num_clust, start_clust, true_p_clust1, true_p_clust2, true_p_clust3, p_axis

def cluster_dedx_reco_dir(voxels, values, start, reco_dir, dedx_dist=3):
    assert voxels.shape[1] == 3, (
            "The shape of the input is not compatible with voxel coordinates.")
    # If start point is not in voxels, assign the closest point within voxels
    # as the startpoint
    if not np.isclose(start, voxels, atol=1e-2).all(axis=1).any():
        dists = np.linalg.norm(voxels - start, axis=1)
        perm = np.argsort(dists)
        start = voxels[perm[0]]
    # distance from the startpoint
    dist_mat = cdist(start.reshape(1,-1), voxels).flatten()
    # legacy dedx
    #if dedx_dist > 0:
    voxels_dedx = voxels[dist_mat <= dedx_dist]
    values_dedx = values[dist_mat <= dedx_dist]
    dist_dedx = dist_mat[dist_mat <= dedx_dist]

    if len(voxels_dedx) < 2:
        return 0., 0., 0., len(voxels_dedx)

    end_pt = start+dedx_dist*reco_dir
    p_voxels = np.dot(voxels_dedx - start, reco_dir)
    mask = (p_voxels >= -1e-2) & (p_voxels <= 3)
    #print("start :", start)
    #print("reco dir :", reco_dir)
    #print("end_pt ", end_pt)

    voxels_de = voxels_dedx[mask]
    values_de = values_dedx[mask]
    #print("values_de: ", values_de)
    if len(voxels_de) < 2:
        return 0., 0., 0., len(voxels_de)
    #print(len(voxels_de), len(voxels_dedx))
    
    p_voxels_de = np.dot(voxels_de - start, reco_dir)
    dx = -min(p_voxels_de)+max(p_voxels_de)
    voxels_sp = voxels_de - start
    p_voxels_sp = np.dot(voxels_sp, reco_dir)
    #print("projected voxels_de :", p_voxels_de)
    vectors_to_axis = voxels_sp - np.outer(p_voxels_sp, reco_dir)
    spread = np.linalg.norm(vectors_to_axis, axis=1)
    spread = sum(spread)/len(voxels_sp)
    #print(sum(values_de)/dx, spread)
    return sum(values_de), dx, spread, len(values_de)

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
        self.update_keys({'clust_label_adapt': True, 'meta': True, 'particles': True, 'clust_label_g4': True})
        self.units = 'cm'

    def process(self, data):
        """Evaluate shower start dE/dx for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the keys you want
        print("index: ", data['index'], ", len of truth: ", len(data['truth_particles']))
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
                
            startpoint = true_shower.start_point
            out_dict['true_particle_id'] = true_shower.id
            out_dict['true_particle_pdg'] = true_shower.pdg_code
            out_dict['true_particle_energy_init'] = true_shower.energy_init
            out_dict['true_particle_energy_deposit'] = true_shower.energy_deposit
            out_dict['true_is_contained'] = true_shower.is_contained
            #sum_dedx, max_dist_dedx, p_sum, p_length, p_sum_inc, p_length_inc, spread, p_fit, num_clust, start_clust, true_p_clust1, true_p_clust2, true_p_clust3
            p_de_inc, p_dx_inc, spread, p_fit, num_clust, start_clust_size, true_p_axis, true_clust_sizes  = cluster_dedx_DBScan_PCA(true_shower.points, true_shower.depositions, startpoint, dedx_dist=self.radius, detailed=True)

            true_dedx_DBScan =  cluster_dedx_DBScan_PCA(true_shower.points, true_shower.depositions, startpoint, dedx_dist=self.radius, simple=True)
            print("simple true DBScan PCA: ", true_dedx_DBScan)

            print(true_p_axis)
            print(true_clust_sizes)
                        
            #out_dict['true_de_1'] = t_de_1
            #out_dict['true_dx_1'] = t_dx_1
            #out_dict['true_PCA_de'] = p_de
            #out_dict['true_PCA_dx'] = p_dx
            out_dict['true_PCA_de_inc'] = p_de_inc
            out_dict['true_PCA_dx_inc'] = p_dx_inc
            out_dict['true_p_spread'] = spread
            out_dict['true_p_fit'] = p_fit
            out_dict['true_p_num_clust'] = num_clust
            #out_dict['true_l_1'] = t_l_1
            out_dict['true_match_overlap'] = -1
            out_dict['true_p_start_clust_sisze'] = start_clust_size
            #out_dict['true_p_clust_size1'] = true_p_clusts0
            #out_dict['true_p_clust_size2'] = true_p_clusts1
            #out_dict['true_p_clust_size3'] = true_p_clusts2

            #out_dict['true_p_axis_x'] = true_p_axis[0]
            #out_dict['true_p_axis_y'] = true_p_axis[1]
            #out_dict['true_p_axis_z'] = true_p_axis[2]

            true_p_dir = true_shower.momentum/np.linalg.norm(true_shower.momentum)
            out_dict['true_dir_x'] = true_p_dir[0]
            out_dict['true_dir_y'] = true_p_dir[1]
            out_dict['true_dir_z'] = true_p_dir[2]

            true_de_dir, true_dx_dir, true_spread_dir, true_size_dir =  cluster_dedx_dir(true_shower.points, true_shower.depositions, startpoint, true_p_dir)
            out_dict['true_de_dir'] = true_de_dir
            out_dict['true_dx_dir'] = true_dx_dir
            out_dict['true_spread_dir'] = true_spread_dir
            out_dict['true_size_dir'] = true_size_dir
            
            sed_points = data['clust_label_g4'][:,1:4]
            sed_points = data['meta'].to_cm(sed_points)
            sed_vals = data['clust_label_g4'][:,4]
            #sed_t_de_1, sed_t_dx_1, sed_p_de, sed_p_dx, sed_p_de_inc, sed_p_dx_inc, sed_spread, sed_p_fit, sed_num_clust, sed_start_clust, sed_true_p_clusts0, sed_true_p_clusts1, sed_true_p_clusts2, sed_true_p_axis  = cluster_dedx2_with_PCA(sed_points, sed_vals, startpoint, dedx_dist=self.radius)

            #out_dict['sed_true_de_1'] = sed_t_de_1
            #out_dict['sed_true_dx_1'] = sed_t_dx_1
            #out_dict['sed_true_PCA_de'] = sed_p_de
            #out_dict['sed_true_PCA_dx'] = sed_p_dx
            #out_dict['sed_true_PCA_de_inc'] = sed_p_de_inc
            #out_dict['sed_true_PCA_dx_inc'] = sed_p_dx_inc
            #out_dict['sed_true_p_spread'] = sed_spread
            #out_dict['sed_true_p_fit'] = sed_p_fit
            #out_dict['sed_true_p_num_clust'] = sed_num_clust
            #out_dict['true_l_1'] = t_l_1                                                                                                                               
            #out_dict['sed_true_p_start_clust'] = sed_start_clust
            #out_dict['sed_true_p_clust_size1'] = sed_true_p_clusts0
            #out_dict['sed_true_p_clust_size2'] = sed_true_p_clusts1
            #out_dict['sed_true_p_clust_size3'] = sed_true_p_clusts2
            #out_dict['sed_true_p_axis_x'] = sed_true_p_axis[0]
            #out_dict['sed_true_p_axis_y'] = sed_true_p_axis[1]
            #out_dict['sed_true_p_axis_z'] = sed_true_p_axis[2]

            sed_de_dir, sed_dx_dir, sed_spread_dir, sed_size_dir =  cluster_dedx_dir(sed_points, sed_vals, startpoint, true_p_dir)
            out_dict['sed_de_dir'] = sed_de_dir
            out_dict['sed_dx_dir'] = sed_dx_dir
            out_dict['sed_spread_dir'] = sed_spread_dir
            out_dict['sed_size_dir'] = sed_size_dir
            
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

            reco_dir = reco_shower.start_dir
            #print("reco_dir: ", reco_dir)
            #print("reco_startpint:", reco_shower.start_point)
            out_dict['reco_dir_x'] = reco_dir[0]
            out_dict['reco_dir_y'] = reco_dir[1]
            out_dict['reco_dir_z'] = reco_dir[2]
            
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
                        
            de_0, dx_0, l_0 = cluster_dedx_legacy(reco_shower.points, 
                                  reco_shower.depositions,
                                     startpoint, 
                                     max_dist=self.radius)
            dedx_legacy = cluster_dedx_legacy(reco_shower.points,
                                  reco_shower.depositions,
                                     startpoint,
                                                  max_dist=self.radius, simple=True)
            print("simple legacy :", dedx_legacy )
            out_dict['reco_de_0'] = de_0
            out_dict['reco_dx_0'] = dx_0
            out_dict['reco_l_0'] = l_0

            #reco_de_1, reco_dx_1, reco_p_de, reco_p_dx, reco_p_de_inc, reco_p_dx_inc, reco_spread, reco_p_fit, reco_num_clust, reco_start_clust, reco_p_clusts0, reco_p_clusts1, reco_p_clusts2, reco_p_axis  = cluster_dedx2_with_PCA(reco_shower.points, reco_shower.depositions, startpoint, dedx_dist=self.radius)
            #out_dict['reco_de'] = reco_de_1
            #out_dict['reco_dx'] = reco_dx_1
            #out_dict['reco_PCA_de'] = reco_p_de
            #out_dict['reco_PCA_dx'] = reco_p_dx
            #out_dict['reco_PCA_de_inc'] = reco_p_de_inc
            #out_dict['reco_PCA_dx_inc'] = reco_p_dx_inc
            #out_dict['reco_p_spread'] = reco_spread
            #out_dict['reco_p_fit'] = reco_p_fit
            #out_dict['reco_p_num_clust'] = reco_num_clust
            
            #out_dict['reco_p_start_clust'] = reco_start_clust
            #out_dict['reco_p_clust_size1'] = reco_p_clusts0
            #out_dict['reco_p_clust_size2'] = reco_p_clusts1
            #out_dict['reco_p_clust_size3'] = reco_p_clusts2

            #out_dict['reco_p_axis_x'] = reco_p_axis[0]
            #out_dict['reco_p_axis_y'] = reco_p_axis[1]
            #out_dict['reco_p_axis_z'] = reco_p_axis[2]
            #out_dict['reco__0']

            
            reco_de_dir, reco_dx_dir, reco_spread_dir, reco_size_dir =  cluster_dedx_dir(reco_shower.points, reco_shower.depositions, reco_shower.start_point, reco_shower.start_dir)
            out_dict['reco_de_dir'] = reco_de_dir
            out_dict['reco_dx_dir'] = reco_dx_dir
            out_dict['reco_spread_dir'] = reco_spread_dir
            out_dict['reco_size_dir'] = reco_size_dir

            reco_dedx_dir =  cluster_dedx_dir(reco_shower.points, reco_shower.depositions, reco_shower.start_point, reco_shower.start_dir, simple=True)
            print("simple dir: ", reco_dedx_dir)
            
            de_1, dx_1, l_1 = cluster_dedx_legacy(reco_points,
                                  reco_vals,
                                  startpoint,
                                     max_dist=self.radius)
            out_dict['reco_de_1'] = de_1
            out_dict['reco_dx_1'] = dx_1
            out_dict['reco_l_1'] = l_1
            
            # dedx with true startpoint
            de_02, dx_02, l_02 = cluster_dedx_legacy(reco_shower.points, 
                                  reco_shower.depositions, 
                                  true_shower.start_point, 
                                     max_dist=self.radius)
            out_dict['reco_de_02'] = de_02
            out_dict['reco_dx_02'] = dx_02
            out_dict['reco_l_02'] = l_02
            
            de_2, dx_2, l_2 = cluster_dedx_legacy(reco_points,
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

            de_03, dx_03, l_03  = cluster_dedx_legacy(reco_shower.points,                                                                                                                                
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
                de_3, dx_3, l_3 = cluster_dedx_legacy(reco_points,
                                  reco_vals,
                                  closest_point,
                                    max_dist=self.radius)


            out_dict['reco_de_3'] = de_3
            out_dict['reco_dx_3'] = dx_3
            out_dict['reco_l_3'] = l_3
            
            self.append('particle', **out_dict)
