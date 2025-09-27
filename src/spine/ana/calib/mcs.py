"""MCS angular resolution calibration module.

This module uses high energy muon tracks to evaluate the base angular resolution
of the tracks as a basis to correct the scattering angle spread in the MCS-based
KE estimation used in SPINE.
"""

from functools import partial

import numpy as np

from spine.ana.base import AnaBase
from spine.utils.globals import MUON_PID
from spine.utils.mcs import ANGLE_METHODS, mcs_angles, mcs_angles_proj
from spine.utils.tracking import get_track_segments

__all__ = ["MCSCalibAna"]


class MCSCalibAna(AnaBase):
    """This analysis script estimates the angular resolution of high energy
    muon chunks to calibrate the MCS-based KE reconstruction.

    This script proceeds through the following steps:
    - Identify high energy muons
    - Chunk them up in various sizes
    - Estimate the 3D polar angle between successive chunks
    """

    # Name of the analysis script (as specified in the configuration)
    name = "calib_mcs"

    # Axes names used to store direction information
    _axes = ("x", "y", "z")

    # Projection names used to store projected angle information
    _projs = ("yz", "xz", "xy")

    def __init__(
        self,
        min_ke,
        segment_length,
        segment_method="bin_pca",
        anchor_point=True,
        angle_method="atan2",
        time_window=None,
        run_mode="truth",
        truth_point_mode="points",
        **kwargs,
    ):
        """Initialize the analysis script.

        Parameters
        ----------
        min_ke : float
            Kinetic energy (in MeV) above which a muon is included in the estimate
        segment_length : Union[float, List[float]]
            Segment length in the units that specify the coordinates. If array,
            will produce angular measurements for each segmentation length
        segment_method : str, default 'step_next'
            Method used to segment the track (one of 'step', 'step_next'
            or 'bin_pca')
        anchor_point : bool, default True
            Weather or not to collapse end point onto the closest track point
        angle_method : str, default 'atan2'
            Angular reconstruction method
        time_window : List[float], optional
            Time within which the particle must occur to be used for calibration.
            This is only used with true instances to remove out-of-time objects
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(
            "particle", run_mode, truth_point_mode=truth_point_mode, **kwargs
        )

        # Store the kinematic cut
        self.min_ke = min_ke

        # Store the segmentation parameters
        self.segment_length = segment_length
        if np.isscalar(segment_length):
            self.segment_length = [segment_length]

        # Partially initialize the segmentation algorithm
        self.get_track_segments = partial(
            get_track_segments, method=segment_method, anchor_point=anchor_point
        )

        # Store the angle computation parameters
        assert angle_method in ANGLE_METHODS, (
            f"Angular reconstruction method not recognized: {angle_method}. "
            f"Must be one of {ANGLE_METHODS.keys()}"
        )
        self.angle_method = ANGLE_METHODS[angle_method]

        # Store the time window, if provided
        assert (
            time_window is None or run_mode == "truth"
        ), "Cannot enforce time window containment on reconstructed objects."
        assert time_window is None or (
            not np.isscalar(time_window) and len(time_window) == 2
        ), "If a time window is provided, it must be a list of two scalars."
        self.time_window = time_window

        # Initialize the CSV writer(s) you want (one per segment length)
        for prefix in self.prefixes:
            for sl in self.segment_length:
                self.initialize_writer(f"{prefix}_mcs_sl{sl}")

    def process(self, data):
        """Pass data products corresponding to one entry through the analysis.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over particle keys
        for prefix in self.prefixes:
            # Loop over particles
            for part in data[f"{prefix}_particles"]:
                # Check PID and kinetic energy
                if part.pid != MUON_PID or part.ke < self.min_ke:
                    continue

                # Enfore time, if requested
                if self.time_window is not None:
                    if part.t < self.time_window[0] or part.t > self.time_window[1]:
                        continue

                # Loop over segment lengths
                for sl in self.segment_length:
                    # Segment the track
                    segments, dirs, lengths = self.get_track_segments(
                        self.get_points(part), sl, part.start_point
                    )
                    if len(segments) < 1:
                        continue

                    # Compute angles and directions
                    angles = mcs_angles(dirs, self.angle_method)
                    angles_proj = mcs_angles_proj(dirs, self.angle_method)
                    dirs = (dirs[:-1] + dirs[1:]) / 2

                    # Record the size of the smallest of the two segments
                    counts = np.array([len(seg) for seg in segments])
                    min_counts = np.min(np.vstack((counts[:-1], counts[1:])), axis=0)

                    # Record the distance between two successive segment centers
                    distances = (lengths[:-1] + lengths[1:]) / 2

                    # Write to the file
                    file_name = f"{prefix}_mcs_sl{sl}"
                    base = {"ke": part.ke}
                    if part.is_truth:
                        base["time"] = part.t
                    for i in range(len(angles)):
                        dir_dict = {
                            f"dir_{a}": dirs[i, j] for j, a in enumerate(self._axes)
                        }
                        proj_dict = {
                            f"angle_{p}": angles_proj[i, j]
                            for j, p in enumerate(self._projs)
                        }
                        self.append(
                            file_name,
                            **base,
                            **dir_dict,
                            **proj_dict,
                            angle=angles[i],
                            min_count=min_counts[i],
                            distance=distances[i],
                        )
