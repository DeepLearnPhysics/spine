"""Tracking reconstruction modules."""

import numpy as np

from spine.utils.globals import (
        TRACK_SHP, MUON_PID, PION_PID, PROT_PID, KAON_PID)
from spine.utils.energy_loss import csda_table_spline
from spine.utils.tracking import get_track_length

from spine.post.base import PostBase

__all__ = ['CSDAEnergyProcessor']


class CSDAEnergyProcessor(PostBase):
    """Reconstruct the kinetic energy of tracks based on their range in liquid
    argon using the continuous slowing down approximation (CSDA).
    """
    name = 'csda_ke'
    aliases = ['reconstruct_csda_energy']

    def __init__(self, tracking_mode='step_next',
                 include_pids=[MUON_PID, PION_PID, PROT_PID, KAON_PID],
                 obj_type='particle', run_mode='both',
                 truth_point_mode='points', **kwargs):
        """Store the necessary attributes to do CSDA range-based estimation.

        Parameters
        ----------
        tracking_mode : str, default 'step_next'
            Method used to compute the track length (one of 'displacement',
            'step', 'step_next', 'bin_pca' or 'spline')
        include_pids : list, default [2, 3, 4, 5]
            Particle species to compute the kinetic energy for
        **kwargs : dict, optional
            Additional arguments to pass to the tracking algorithm
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode)

        # Fetch the functions that map the range to a KE
        self.include_pids = include_pids
        self.splines = {
                ptype: csda_table_spline(ptype) for ptype in include_pids}

        # Store the tracking parameters
        self.tracking_mode = tracking_mode
        self.tracking_kwargs = kwargs

    def process(self, data):
        """Reconstruct the CSDA KE estimates for each particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle objects
        for k in self.fragment_keys + self.particle_keys:
            for obj in data[k]:
                # Only run this algorithm on tracks that have a CSDA table
                if not ((obj.shape == TRACK_SHP) and
                        (obj.pid in self.include_pids)):
                    continue

                # Make sure the object coordinates are expressed in cm
                self.check_units(obj)

                # Get point coordinates
                points = self.get_points(obj)
                if not len(points):
                    continue

                # Compute the length of the track
                length = get_track_length(
                        points, point=obj.start_point,
                        method=self.tracking_mode, **self.tracking_kwargs)

                # Store the length
                if not obj.is_truth:
                    obj.length = length
                else:
                    obj.reco_length = length

                # Compute the CSDA kinetic energy
                if length > 0.:
                    obj.csda_ke = self.splines[obj.pid](length).item()
                else:
                    obj.csda_ke = 0.
