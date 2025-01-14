'''Module to evaluate diagnostic metrics on showers.'''

import numpy as np

from spine.ana.base import AnaBase
from spine.utils.gnn.cluster import cluster_dedx

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
            
            dedx_1 = cluster_dedx(true_shower.points, 
                                  true_shower.depositions, 
                                  startpoint, 
                                  max_dist=self.radius)
            
            dedx_2 = cluster_dedx(true_shower.truth_points, 
                                  true_shower.truth_depositions, 
                                  startpoint, 
                                  max_dist=self.radius)
            
            out_dict['true_dedx_1'] = dedx_1
            out_dict['true_dedx_2'] = dedx_2
            
            # Reco Showers, reco points
            
            if len(data['reco_particles']) == 0:
                return
            
            reco_shower = data['reco_particles'][0]
            startpoint = reco_shower.start_point
            out_dict['reco_particle_id'] = reco_shower.id
            # out_dict['reco_particle_energy_init'] = reco_shower.energy_init
            out_dict['reco_particle_energy_deposit'] = reco_shower.calo_ke
            out_dict['reco_is_contained'] = reco_shower.is_contained
            
            dedx_1 = cluster_dedx(reco_shower.points, 
                                  reco_shower.depositions, 
                                  startpoint, 
                                  max_dist=self.radius)
            out_dict['reco_dedx_1'] = dedx_1
            
            # dedx with true startpoint
            dedx_2 = cluster_dedx(reco_shower.points, 
                                  reco_shower.depositions, 
                                  true_shower.start_point, 
                                  max_dist=self.radius)
            
            # dedx with point closest to true startpoint
            
            dists = np.linalg.norm(reco_shower.points - true_shower.start_point, axis=1)
            perm = np.argsort(dists)
            closest_point = reco_shower.points[perm[0]]
            
            dedx_3 = cluster_dedx(reco_shower.points,
                                  reco_shower.depositions,
                                  closest_point,
                                  max_dist=self.radius)
            
            out_dict['reco_dedx_2'] = dedx_2
            out_dict['reco_dedx_3'] = dedx_3
            
            self.append('particle', **out_dict)
