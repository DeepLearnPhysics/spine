"""Analysis module template.

Evaluate performance of the fmatch algorithm for SBND.
"""

# Add the imports specific to this module here
import numpy as np

# Must import the analysis script base class
from spine.ana.base import AnaBase

# Must list the analysis script(s) here to be found by the factory.
# You must also add it to the list of imported modules in the
# `spine.ana.factories`!
__all__ = ['FMatchPerformance']


class FMatchPerformance(AnaBase):
    """
    Evaluate performance of the fmatch algorithm for SBND.
    """

    # Name of the analysis script (as specified in the configuration)
    name = 'fmatch_performance'

    def __init__(self, log_name, flash_tmin=-0.4, flash_tmax=1.5, flash_tolerance=0.5, **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        log_name : str
            name of the analysis script
        flash_tmin : float, optional
            Minimum flash time to be within beam window, by default -0.4 (us)
        flash_tmax : float, optional
            Maximum flash time to be within beam window, by default 1.5 (us)
        flash_tolerance : float, optional
            Tolerance for flash time, by default 0.5 (us)
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Initialize the CSV writer(s) you want
        self.log_name = log_name
        self.initialize_writer(f'{log_name}_fmatch_performance')

        # Add additional required data products
        self.update_keys({'interaction_matches_t2r': True}) 
        self.update_keys({'interaction_matches_t2r_overlap': True}) 

        self.update_keys({'interaction_matches_r2t': True}) 
        self.update_keys({'interaction_matches_r2t_overlap': True}) 
        
        # Initialize the flash time window
        self.flash_tmin = flash_tmin
        self.flash_tmax = flash_tmax
        self.flash_tolerance = flash_tolerance

    def process(self, data):
        """Pass data products corresponding to one entry through the analysis.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the keys you want
        interaction_matches_r2t = data['interaction_matches_r2t']
        interaction_matches_t2r = data['interaction_matches_t2r']
        
        # Loop over matched interactions (t2r)
        for idx, (true_inter, reco_inter) in enumerate(interaction_matches_t2r):
        #for idx, (reco_inter, true_inter) in enumerate(interaction_matches_r2t):
            #if true_inter == None: continue
            #if reco_inter == None: continue
            
            # Storage
            row_dict = {}
            
            
            # Basic info - reco
            if reco_inter == None:
                row_dict['reco_interaction_id'] = -1
                row_dict['reco_is_contained'] = False
                row_dict['reco_is_fiducial'] = False
                row_dict['reco_vtx_x'] = -9999
                row_dict['reco_vtx_y'] = -9999
                row_dict['reco_vtx_z'] = -9999
                row_dict['reco_particle_count0'] = -1
                row_dict['reco_particle_count1'] = -1
                row_dict['reco_particle_count2'] = -1
                row_dict['reco_particle_count3'] = -1
                row_dict['reco_particle_count4'] = -1
                row_dict['reco_flash_volume_id0'] = -1
                row_dict['reco_flash_time0'] = -9999
                row_dict['reco_flash_id0'] = -1
                row_dict['reco_flash_score0'] = -9999
                row_dict['reco_flash_volume_id1'] = -1
                row_dict['reco_flash_time1'] = -9999
                row_dict['reco_flash_id1'] = -1
                row_dict['reco_flash_score1'] = -9999
                row_dict['reco_flash_total_pe'] = -1
                row_dict['reco_flash_hypo_pe'] = -1
                row_dict['reco_reduced_flash_score'] = -9999
                row_dict['reco_calo_energy'] = -1
                row_dict['reco_in_bnb'] = False
            else:
                row_dict['reco_interaction_id'] = reco_inter.id
                row_dict['reco_is_contained'] = reco_inter.is_contained
                row_dict['reco_is_fiducial'] = reco_inter.is_fiducial
                row_dict['reco_vtx_x'] = reco_inter.vertex[0]
                row_dict['reco_vtx_y'] = reco_inter.vertex[1]
                row_dict['reco_vtx_z'] = reco_inter.vertex[2]
                row_dict['reco_particle_count0'] = reco_inter.particle_counts[0]
                row_dict['reco_particle_count1'] = reco_inter.particle_counts[1]
                row_dict['reco_particle_count2'] = reco_inter.particle_counts[2]
                row_dict['reco_particle_count3'] = reco_inter.particle_counts[3]
                row_dict['reco_particle_count4'] = reco_inter.particle_counts[4]
                
                #Initialize flash info
                for i in range(2): #2 volumes
                    #Reco
                    row_dict[f'reco_flash_volume_id{i}'] = -1
                    row_dict[f'reco_flash_time{i}'] = -9999
                    row_dict[f'reco_flash_id{i}'] = -1
                    row_dict[f'reco_flash_score{i}'] = -9999
                    
                # Reco flash info
                for i, fid in enumerate(reco_inter.flash_volume_ids):
                    row_dict[f'reco_flash_volume_id{fid}'] = fid
                    row_dict[f'reco_flash_time{fid}'] = reco_inter.flash_times[i]
                    row_dict[f'reco_flash_id{fid}'] = reco_inter.flash_ids[i]
                    row_dict[f'reco_flash_score{fid}'] = reco_inter.flash_scores[i]
                
                row_dict[f'reco_flash_total_pe'] = reco_inter.flash_total_pe
                row_dict[f'reco_flash_hypo_pe'] = reco_inter.flash_hypo_pe
                row_dict[f'reco_reduced_flash_score'] = (reco_inter.flash_total_pe - reco_inter.flash_hypo_pe) / reco_inter.flash_total_pe
                
                # Energy info - get KE of all reco particles
                row_dict['reco_calo_energy'] = 0
                for i,p in enumerate(reco_inter.particles):
                    row_dict[f'reco_calo_energy'] += p.calo_ke
                
                if (row_dict['reco_flash_time0'] < (self.flash_tmax + self.flash_tolerance) and row_dict['reco_flash_time0'] > (self.flash_tmin - self.flash_tolerance))\
                    or (row_dict['reco_flash_time1'] < (self.flash_tmax + self.flash_tolerance) and row_dict['reco_flash_time1'] > (self.flash_tmin - self.flash_tolerance)):
                    row_dict['reco_in_bnb'] = True
                else:    
                    row_dict['reco_in_bnb'] = False
                    
                
            
            # Basic info - truth
            row_dict['truth_interaction_id'] = true_inter.id
            row_dict['truth_is_contained'] = true_inter.is_contained
            row_dict['truth_is_fiducial'] = true_inter.is_fiducial
            row_dict['truth_vtx_x'] = true_inter.vertex[0]
            row_dict['truth_vtx_y'] = true_inter.vertex[1]
            row_dict['truth_vtx_z'] = true_inter.vertex[2]
            row_dict['truth_particle_count0'] = true_inter.particle_counts[0]
            row_dict['truth_particle_count1'] = true_inter.particle_counts[1]
            row_dict['truth_particle_count2'] = true_inter.particle_counts[2]
            row_dict['truth_particle_count3'] = true_inter.particle_counts[3]
            row_dict['truth_particle_count4'] = true_inter.particle_counts[4]
            row_dict['truth_iscc'] = true_inter.current_type == 0
            row_dict['truth_nu_id'] = true_inter.nu_id
            
            # Flash info
            for i in range(2): #2 volumes
                #Truth
                row_dict[f'truth_flash_volume_id{i}'] = -1
                row_dict[f'truth_flash_time{i}'] = -9999
                row_dict[f'truth_flash_id{i}'] = -1
                row_dict[f'truth_flash_score{i}'] = -9999
            
            # Truth flash info
            for i, fid in enumerate(true_inter.flash_volume_ids):
                row_dict[f'truth_flash_volume_id{fid}'] = fid
                row_dict[f'truth_flash_time{fid}'] = true_inter.flash_times[i]
                row_dict[f'truth_flash_id{fid}'] = true_inter.flash_ids[i]
                #Check if the index for this score exists
                if i < len(true_inter.flash_scores):
                    row_dict[f'true_flash_score{fid}'] = true_inter.flash_scores[i]
                row_dict[f'truth_flash_score{fid}'] = true_inter.flash_scores[i]
            
            row_dict[f'truth_flash_total_pe'] = true_inter.flash_total_pe
            row_dict[f'truth_flash_hypo_pe'] = true_inter.flash_hypo_pe
            row_dict[f'truth_reduced_flash_score'] = (true_inter.flash_total_pe - true_inter.flash_hypo_pe) / true_inter.flash_total_pe
                
            #Energy info - get E of all true particles
            row_dict['truth_energy_init'] = 0
            # Energy info - get KE of all true particles
            row_dict['truth_calo_energy'] = 0
            # Also get average time of particles
            row_dict['truth_avg_time'] = 0
            
            nprim = 0
            for i,p in enumerate(true_inter.particles):
                row_dict[f'truth_calo_energy'] += p.calo_ke
                row_dict['truth_energy_init'] += p.energy_init
                if p.is_primary:
                    row_dict[f'truth_avg_time'] += p.t*1e-3 #us
                    nprim += 1
            if nprim > 0:
                row_dict['truth_avg_time'] /= nprim
            else:
                row_dict['truth_avg_time'] = -9999
            
            #Efficiency - is the flash time within the BNB window?
            # Truth values get some slack because of the time resolution +- 0.2 us
            if row_dict['truth_avg_time'] < (self.flash_tmax + self.flash_tolerance) and row_dict['truth_avg_time'] > (self.flash_tmin - self.flash_tolerance):
                row_dict['truth_in_bnb'] = True
            else:
                row_dict['truth_in_bnb'] = False
            
            # Overlap
            #overlap = data['interaction_matches_t2r_overlap'][idx]
            #overlap = data['interaction_matches_r2t_overlap'][idx]
            #row_dict.update({'match_overlap': overlap})
            self.append(f'{self.log_name}_fmatch_performance', **row_dict)