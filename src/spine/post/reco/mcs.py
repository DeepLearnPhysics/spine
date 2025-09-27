"""Multiple-Coulomb scattering (MCS) energy reconstruction module."""

import numpy as np

from spine.post.base import PostBase
from spine.utils.globals import (
    KAON_PID,
    MUON_PID,
    PID_MASSES,
    PION_PID,
    PROT_PID,
    TRACK_SHP,
)
from spine.utils.mcs import ANGLE_METHODS, mcs_angles, mcs_fit
from spine.utils.tracking import get_track_segments

__all__ = ["MCSEnergyProcessor"]


class MCSEnergyProcessor(PostBase):
    """Reconstruct the kinetic energy of tracks based on their Multiple-Coulomb
    scattering (MCS) angles while passing through liquid argon.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "mcs_ke"

    # Alternative allowed names of the post-processor
    aliases = ("reconstruct_mcs_energy",)

    def __init__(
        self,
        tracking_mode="bin_pca",
        segment_length=5.0,
        res_a=0.25,
        res_b=1.25,
        res_mixture=False,
        res_weight_ratio=0.5,
        res_scale_ratio=2.25,
        include_pids=(MUON_PID, PION_PID, PROT_PID, KAON_PID),
        fill_per_pid=False,
        only_uncontained=False,
        angle_method="atan2",
        obj_type="particle",
        run_mode="both",
        truth_point_mode="points",
        pid_mode="pid",
        **kwargs,
    ):
        """Store the necessary attributes to do MCS-based estimations.

        Parameters
        ----------
        tracking_mode : str, default 'bin_pca'
            Method used to compute the segment angles (one of 'step',
            'step_next' or 'bin_pca')
        segment_length : float, default 5.0 cm
            Segment length in the units that specify the coordinates
        res_a : float, default 0.25 rad*cm^res_b
            Parameter a in the a/dx^b which models the angular uncertainty
        res_b : float, default 1.25
            Parameter b in the a/dx^b which models the angular uncertainty
        res_mixture : bool, default False
            If `True`, the detector resolution is modeled by a Rayleigh mixture
        res_weight_ratio : float, default 0.5
            When using a Rayleigh mixture, defines the weight ratio between components
        res_scale_ratio : float, default 2.25
            When using a Rayleigh mixture, defines the scale ratio between components
        include_pids : list, default [2, 3, 4, 5]
            Particle species to compute the kinetic energy for
        fill_per_pid : bool, default False
            If `True`, compute the MCS KE estimate under all PID assumptions
        only_uncontained : bool, default False
            Only run the algorithm on particles that are not contained
        angle_method : str, default 'atan2'
            Angular reconstruction method
        **kwargs : dict, optiona
            Additional arguments to pass to the tracking algorithm
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, truth_point_mode, pid_mode=pid_mode)

        # Store the general parameters
        self.include_pids = include_pids
        self.fill_per_pid = fill_per_pid
        self.only_uncontained = only_uncontained

        # Store the tracking parameters
        assert tracking_mode in [
            "step",
            "step_next",
            "bin_pca",
        ], "The tracking algorithm must provide segment angles"
        self.tracking_mode = tracking_mode
        self.tracking_kwargs = kwargs

        # Store the angle computation parameters
        assert angle_method in ANGLE_METHODS, (
            f"Angular reconstruction method not recognized: {angle_method}. "
            f"Must be one of {ANGLE_METHODS.keys()}"
        )
        self.angle_method = ANGLE_METHODS[angle_method]

        # Store the MCS parameters
        self.segment_length = segment_length
        self.res_a = res_a
        self.res_b = res_b
        self.res_mixture = res_mixture
        self.res_weight_ratio = res_weight_ratio
        self.res_scale_ratio = res_scale_ratio

    def process(self, data):
        """Reconstruct the MCS KE estimates for each particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle objects
        for k in self.fragment_keys + self.particle_keys:
            for obj in data[k]:
                # Only run this algorithm on particle species that are needed
                if not (
                    (obj.shape == TRACK_SHP)
                    and (self.get_pid(obj) in self.include_pids)
                ):
                    continue
                if self.only_uncontained and obj.is_contained:
                    continue

                # Make sure the particle coordinates are expressed in cm
                self.check_units(obj)

                # Get point coordinates
                points = self.get_points(obj)
                if not len(points):
                    continue

                # Get the list of segment directions
                _, dirs, _ = get_track_segments(
                    points,
                    self.segment_length,
                    obj.start_point,
                    method=self.tracking_mode,
                    **self.tracking_kwargs,
                )

                # Find the angles between successive segments
                theta = mcs_angles(dirs, self.angle_method)
                if len(theta) < 1:
                    continue

                # Store the length and the MCS kinetic energy
                mass = PID_MASSES[self.get_pid(obj)]
                obj.mcs_ke = mcs_fit(
                    theta,
                    mass,
                    self.segment_length,
                    1,
                    self.res_a,
                    self.res_b,
                    self.res_mixture,
                    self.res_weight_ratio,
                    self.res_scale_ratio,
                )

                # If requested, convert the KE to other PID hypotheses
                if self.fill_per_pid:
                    # Compute the momentum (what MCS is truly sensitive to)
                    mom = np.sqrt(obj.mcs_ke**2 + 2 * mass * obj.mcs_ke)

                    # For each PID, convert back to KE
                    for pid in self.include_pids:
                        mass = PID_MASSES[pid]
                        ke = np.sqrt(mom**2 + mass**2) - mass
                        obj.mcs_ke_per_pid[pid] = ke
