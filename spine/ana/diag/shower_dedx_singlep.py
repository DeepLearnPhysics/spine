'''Module to evaluate diagnostic metrics on showers.'''

import numpy as np

from spine.ana.base import AnaBase
from spine.utils.gnn.cluster import cluster_dedx2, cluster_end_points

__all__ = ['ShowerStartSingleParticle']


class ShowerStartSingleParticle(AnaBase):
    """This analysis script computes the dE/dx value within some distance
    from the start point of an EM shower object.

    This is a useful diagnostic tool to evaluate the calorimetric separability
    of different EM shower types (electron vs photon), which are expected to
    have different dE/dx patterns near their start point.
    """

    # Name of the analysis script (as specified in the configuration)
    name = 'shower_start_singlep'

    def __init__(self, radius, **kwargs):
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

        # Initialize the CSV writer(s) you want
        for obj in self.obj_type:
            self.initialize_writer(obj)
        self.update_keys({'clust_label_adapt': True, 'meta': True})
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
            print(data['index'])
            out_dict['file_index'] = data['file_index']
            out_dict['file_entry_index'] = data['file_entry_index']
            
            # True Showers
            
            true_shower = data['truth_particles'][0]
            startpoint = true_shower.start_point
            out_dict['true_particle_id'] = true_shower.id
            out_dict['true_particle_energy_init'] = true_shower.energy_init
            out_dict['true_particle_energy_deposit'] = true_shower.energy_deposit
            out_dict['true_is_contained'] = true_shower.is_contained
            
            de_1, dx_1 = cluster_dedx2(true_shower.points, 
                                  true_shower.depositions, 
                                  startpoint, 
                                  max_dist=self.radius)
            
                        
            out_dict['true_de_1'] = de_1
            out_dict['true_dx_1'] = dx_1
            out_dict['true_match_overlap'] = -1
            if true_shower.match_overlaps is not None and len(true_shower.match_overlaps)>0:
                out_dict['true_match_overlap'] = true_shower.match_overlaps[0]
            # Reco Showers, reco points
            
            if len(data['reco_particles']) == 0:
                return
            
            reco_shower = data['reco_particles'][0]
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
            #        print(i, reco_vals[i])
            reco_points=data['meta'].to_cm(reco_points)
                        
            de_0, dx_0 = cluster_dedx2(reco_shower.points, 
                                  reco_shower.depositions,
                                     startpoint, 
                                     max_dist=self.radius)
            out_dict['reco_de_0'] = de_0
            out_dict['reco_dx_0'] = dx_0
            
            de_1, dx_1 = cluster_dedx2(reco_points,
                                  reco_vals,
                                  startpoint,
                                     max_dist=self.radius)
            out_dict['reco_de_1'] = de_1
            out_dict['reco_dx_1'] = dx_1
            
            # dedx with true startpoint
            de_02, dx_02 = cluster_dedx2(reco_shower.points, 
                                  reco_shower.depositions, 
                                  true_shower.start_point, 
                                     max_dist=self.radius)
            out_dict['reco_de_02'] = de_02
            out_dict['reco_dx_02'] = dx_02
            
            de_2, dx_2 = cluster_dedx2(reco_points,
                                  reco_vals,
                                  true_shower.start_point,
                                     max_dist=self.radius)
            
            # dedx with point closest to true startpoint
            
            dists = np.linalg.norm(reco_shower.points - true_shower.start_point, axis=1)
            perm = np.argsort(dists)
            closest_point = reco_shower.points[perm[0]]

            de_03, dx_03 = cluster_dedx2(reco_shower.points,                                                                                                                                
                       reco_shower.depositions,                                                                                                                                
                      closest_point,                                                                                                                                          
                       max_dist=self.radius)
            out_dict['reco_de_03'] = de_03
            out_dict['reco_dx_03'] = dx_03
            
            de_3, dx_3 = 0., 0.
            if len(reco_points)>0.:
                dists = np.linalg.norm(reco_points - true_shower.start_point, axis=1)
                perm = np.argsort(dists)
                closest_point = reco_points[perm[0]]

                #de_3, dx_3 = cluster_dedx2(reco_shower.points,
                #                      reco_shower.depositions,
                #                      closest_point,
                #                        max_dist=self.radius)
                de_3, dx_3 = cluster_dedx2(reco_points,
                                  reco_vals,
                                  closest_point,
                                    max_dist=self.radius)
            
            out_dict['reco_de_2'] = de_2
            out_dict['reco_dx_2'] = dx_2
            out_dict['reco_de_3'] = de_3
            out_dict['reco_dx_3'] = dx_3
            
            out_dict['true_startpoint_x'] = true_shower.start_point[0]
            out_dict['true_startpoint_y'] = true_shower.start_point[1]
            out_dict['true_startpoint_z'] = true_shower.start_point[2]
            
            out_dict['reco_startpoint_x'] = reco_shower.start_point[0]
            out_dict['reco_startpoint_y'] = reco_shower.start_point[1]
            out_dict['reco_startpoint_z'] = reco_shower.start_point[2]
            
            # Compute startpoint using umbrella curvature
            
            if reco_points.shape[0] > 3:
                umb_semi_start, _ = cluster_end_points(reco_points)
            else:
                umb_semi_start = np.array([-np.inf, -np.inf, -np.inf])
                
            if reco_shower.points.shape[0] > 3:
                umb_reco_start, _ = cluster_end_points(reco_shower.points)
            else:
                umb_reco_start = np.array([-np.inf, -np.inf, -np.inf])
            
            if true_shower.points.shape[0] > 3:
                umb_true_start, _ = cluster_end_points(true_shower.points)
            else:
                umb_true_start = np.array([-np.inf, -np.inf, -np.inf])
            
            out_dict['umb_semi_start_x'] = umb_semi_start[0]
            out_dict['umb_semi_start_y'] = umb_semi_start[1]
            out_dict['umb_semi_start_z'] = umb_semi_start[2]
            
            out_dict['umb_reco_start_x'] = umb_reco_start[0]
            out_dict['umb_reco_start_y'] = umb_reco_start[1]
            out_dict['umb_reco_start_z'] = umb_reco_start[2]
            
            out_dict['umb_true_start_x'] = umb_true_start[0]
            out_dict['umb_true_start_y'] = umb_true_start[1]
            out_dict['umb_true_start_z'] = umb_true_start[2]
            
            self.append('particle', **out_dict)

