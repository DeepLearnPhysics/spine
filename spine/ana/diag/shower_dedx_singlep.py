'''Module to evaluate diagnostic metrics on showers.'''

import numpy as np

from spine.ana.base import AnaBase
from spine.utils.gnn.cluster import cluster_dedx2

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
            
            t_de_1, t_dx_1, t_l_1 = cluster_dedx2(true_shower.points, 
                                  true_shower.depositions, 
                                  startpoint, 
                                  max_dist=self.radius)
            
                        
            out_dict['true_de_1'] = t_de_1
            out_dict['true_dx_1'] = t_dx_1
            out_dict['true_l_1'] = t_l_1
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
            #     0;276;0c11;rgb:1111/1111/11110;276;0c11;rgb:1111/1111/11110;276;0c11;rgb:1111/1111/11110;276;0c11;rgb:1111/1111/11110;276;0c11;rgb:1111/1111/1111print(i, reco_vals[i])
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
