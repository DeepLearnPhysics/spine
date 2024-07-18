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
    name = 'calo_ke'
    aliases = ['reconstruct_calo_energy']

    def __init__(self, scaling=1., shower_fudge=1., obj_type='particle',
                 run_mode='reco', truth_deposition_mode='points'):
        """Stores the ADC to MeV conversion factor.
        
        Parameters
        ----------
        scaling : Union[float, str]
            Global scaling factor for the depositions (can be an expression)
        shower_fudge : Union[float, str]
            Shower energy fudge factor (accounts for missing cluster energy)
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode)

        # Store parameter
        self.truth_deposition_mode = truth_deposition_mode

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
                if not part.is_truth:
                    part.calo_ke = scaling * part.depositions_sum
                else:
                    part.calo_ke = scaling * getattr(
                            part, self.truth_deposition_mode)


class CalibrationProcessor(PostBase):
    """Apply calibrations to the reconstructed objects."""
    name = 'calibration'
    aliases = ['apply_calibrations']
    keys = {'depositions': True, 'run_info': False}

    def __init__(self, dedx=2.2, do_tracking=False, obj_type='particle', 
                 run_mode='both', **cfg):
        """Initialize the calibration manager.

        Parameters
        ----------
        dedx : float, default 2.2
            Static value of dE/dx used to compute the recombination factor
        do_tracking : bool, default False
            Segment track to get a proper local dQ/dx estimate
        **cfg : dict
            Calibration manager configuration
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode)

        # Initialize the calibrator
        self.calibrator = CalibrationManager(**cfg)
        self.dedx = dedx
        self.do_tracking = do_tracking

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
        for k in self.obj_keys:
            for part in data[k]:
                # Make sure the particle coordinates are expressed in cm
                self.check_units(part)

                # Get point coordinates
                points = self.get_points(part)
                if not len(points):
                    continue

                # Apply calibration
                if not self.do_tracking or part.shape != TRACK_SHP:
                    depositions = self.calibrator(
                            points, part.depositions, part.sources,
                            run_id, self.dedx)
                else:
                    depositions = self.calibrator.process(
                            points, part.depositions, part.sources,
                            run_id, track=True)

                # Update the particle *and* the reference tensor
                part.depositions = depositions
                data['depositions'][part.index] = depositions
