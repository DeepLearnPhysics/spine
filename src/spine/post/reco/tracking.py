"""Tracking reconstruction modules."""

import numpy as np
from scipy.spatial.distance import cdist

from spine.post.base import PostBase
from spine.utils.energy_loss import csda_table_spline
from spine.utils.globals import (
    KAON_PID,
    MUON_PID,
    PION_PID,
    PROT_PID,
    SHOWR_SHP,
    TRACK_SHP,
)
from spine.utils.tracking import get_track_length

__all__ = ["CSDAEnergyProcessor"]


class CSDAEnergyProcessor(PostBase):
    """Reconstruct the kinetic energy of tracks based on their range in liquid
    argon using the continuous slowing down approximation (CSDA).
    """

    # Name of the post-processor (as specified in the configuration)
    name = "csda_ke"

    # Alternative allowed names of the post-processor
    aliases = ("reconstruct_csda_energy",)

    def __init__(
        self,
        tracking_mode="step_next",
        include_pids=(MUON_PID, PION_PID, PROT_PID, KAON_PID),
        fill_per_pid=False,
        obj_type="particle",
        run_mode="both",
        truth_point_mode="points",
        pid_mode="pid",
        **kwargs,
    ):
        """Store the necessary attributes to do CSDA range-based estimation.

        Parameters
        ----------
        tracking_mode : str, default 'step_next'
            Method used to compute the track length (one of 'displacement',
            'step', 'step_next', 'bin_pca' or 'spline')
        include_pids : list, default [2, 3, 4, 5]
            Particle species to compute the kinetic energy for
        fill_per_pid : bool, default False
            If `True`, compute the CSDA KE estimate under all PID assumptions
        **kwargs : dict, optional
            Additional arguments to pass to the tracking algorithm
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode, pid_mode=pid_mode)

        # Fetch the functions that map the range to a KE
        self.include_pids = include_pids
        self.fill_per_pid = fill_per_pid
        self.splines = {ptype: csda_table_spline(ptype) for ptype in include_pids}

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
                if not (
                    (obj.shape == TRACK_SHP)
                    and (self.get_pid(obj) in self.include_pids)
                ):
                    continue

                # Make sure the object coordinates are expressed in cm
                self.check_units(obj)

                # Get point coordinates
                points = self.get_points(obj)
                if not len(points):
                    continue

                # Compute the length of the track
                length = get_track_length(
                    points,
                    point=obj.start_point,
                    method=self.tracking_mode,
                    **self.tracking_kwargs,
                )

                # Store the length
                if not obj.is_truth:
                    obj.length = length
                else:
                    obj.reco_length = length

                # Compute the CSDA kinetic energy
                if length > 0.0:
                    obj.csda_ke = self.splines[self.get_pid(obj)](length).item()
                    if self.fill_per_pid:
                        for pid in self.include_pids:
                            obj.csda_ke_per_pid[pid] = self.splines[pid](length).item()
                else:
                    obj.csda_ke = 0.0
                    if self.fill_per_pid:
                        for pid in self.include_pids:
                            obj.csda_ke_per_pid[pid] = 0.0
