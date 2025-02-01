"""Calorimetric energy reconstruction module."""

import numpy as np

from spine.utils.globals import TRACK_SHP
from spine.utils.calib import CalibrationManager

from spine.post.base import PostBase

__all__ = ['CalorimetricEnergyProcessor', 'CalibrationProcessor']


class CalorimetricEnergyProcessor(PostBase):
    """Compute calorimetric energy by summing the charge depositions and
    scaling by the ADC to MeV conversion factor, if needed.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'calo_ke'

    # Alternative allowed names of the post-processor
    aliases = ('reconstruct_calo_energy',)

    def __init__(self, scaling=1., shower_fudge=1., obj_type='particle',
                 run_mode='reco', truth_dep_mode='depositions'):
        """Stores the ADC to MeV conversion factor.

        Parameters
        ----------
        scaling : Union[float, str], default 1.
            Global scaling factor for the depositions (can be an expression)
        shower_fudge : Union[float, str], default 1.
            Shower energy fudge factor (accounts for missing cluster energy)
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_dep_mode=truth_dep_mode)

        # Store the conversion factor
        self.scaling = scaling
        if isinstance(self.scaling, str):
            self.scaling = eval(self.scaling)

        # Store the shower fudge factor
        self.shower_fudge = shower_fudge
        if isinstance(self.shower_fudge, str):
            self.shower_fudge = eval(self.shower_fudge)

    def process(self, data):
        """Reconstruct the calorimetric KE for each particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle types
        for k in self.obj_keys:
            # Loop over particles
            for part in data[k]:
                # Set up the scaling
                scaling = self.scaling
                if part.shape != TRACK_SHP:
                    scaling *= self.shower_fudge

                # Save the calorimetric energy
                depositions = self.get_depositions(part)
                part.calo_ke = scaling * np.sum(depositions)


class CalibrationProcessor(PostBase):
    """Apply calibrations to the reconstructed objects."""

    # Name of the post-processor (as specified in the configuration)
    name = 'calibration'

    # Alternative allowed names of the post-processor
    aliases = ('apply_calibrations',)

    # Set of data keys needed for this post-processor to operate
    _keys = (('run_info', False),)

    def __init__(self, do_tracking=False, obj_type=('particle', 'interaction'),
                 run_mode='reco', truth_point_mode='points', **cfg):
        """Initialize the calibration manager.

        Parameters
        ----------
        do_tracking : bool, default False
            Segment track to get a proper local dQ/dx estimate
        **cfg : dict
            Calibration manager configuration
        """
        # Figure out which truth deposition attribute to use
        truth_dep_mode = truth_point_mode.replace('points', 'depositions') + '_q'

        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode, truth_dep_mode)

        # Initialize the calibrator
        self.calibrator = CalibrationManager(**cfg)
        self.do_tracking = do_tracking

        # Add necessary keys
        keys = {}
        if run_mode != 'truth':
            keys.update({
                'points': True,
                'depositions': True,
                'sources': True
            })

        if run_mode != 'reco':
            keys.update({
                self.truth_point_key: True,
                self.truth_dep_key: True,
                self.truth_source_key: True
            })

        self.update_keys(keys)

    def process(self, data):
        """Apply calibrations to each particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the run info
        run_id = None
        if 'run_info' in data:
            run_info = data['run_info']
            run_id = run_info.run

        # Loop over particle objects
        for k in self.particle_keys:
            points_key = 'points' if not 'truth' in k else self.truth_point_key
            source_key = 'sources' if not 'truth' in k else self.truth_source_key
            dep_key = 'depositions' if not 'truth' in k else self.truth_dep_key
            unass_mask = np.ones(len(data[dep_key]), dtype=bool)
            for part in data[k]:
                # Make sure the particle coordinates are expressed in cm
                self.check_units(part)

                # Get point coordinates, sources and depositions
                points = self.get_points(part)
                if not len(points):
                    continue

                sources = self.get_sources(part)
                deps = self.get_depositions(part)

                # Apply calibration
                if not self.do_tracking or part.shape != TRACK_SHP:
                    depositions = self.calibrator(
                            points, deps, sources, run_id)
                else:
                    depositions = self.calibrator.process(
                            points, deps, sources, run_id, track=True)

                # Update the particle *and* the reference tensor
                if not part.is_truth:
                    part.depositions = depositions
                else:
                    setattr(part, self.truth_dep_mode, depositions)

                data[dep_key][part.index] = depositions
                unass_mask[part.index] = False

            # Apply calibration corrections to unassociated depositions
            unass_index = np.where(unass_mask)[0]
            data[dep_key][unass_index] = self.calibrator(
                    data[points_key][unass_index], data[dep_key][unass_index],
                    data[source_key][unass_index], run_id)

        # If requested, updated the depositions attribute of interactions
        for k in self.interaction_keys:
            dep_key = 'depositions' if not 'truth' in k else self.truth_dep_key
            for inter in data[k]:
                # Update depositions for the interaction
                depositions = data[dep_key][inter.index]
                if not part.is_truth:
                    inter.depositions = depositions
                else:
                    setattr(inter, self.truth_dep_mode, depositions)
