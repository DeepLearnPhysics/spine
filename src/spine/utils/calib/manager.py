"""Loads all requested calibration modules and executes them
in the appropriate sequence."""

from copy import deepcopy
from typing import Optional

import numpy as np

from spine.data.meta import Meta
from spine.geo import GeoManager
from spine.utils.stopwatch import StopwatchManager

from .factories import calibrator_factory


class CalibrationManager:
    """Manager in charge of applying all calibration-related corrections onto
    a set of 3D space points and their associated measured charge depositions.
    """

    def __init__(self, gain_applied: bool = False, **cfg):
        """Initialize the manager.

        Parameters
        ----------
        gain_applied : bool, default False
            Whether the gain conversion was applied upstream or not
        **cfg : dict, optional
            Calibrator configurations
        """
        # Fetch the geometry instance
        self.geo = GeoManager.get_instance()

        # Make sure the essential calibration modules are present
        assert (
            gain_applied or "recombination" not in cfg or "gain" in cfg
        ), "Must provide gain configuration if recombination is applied."

        # Add the modules to a processor list in decreasing order of priority
        self.modules = {}
        self.watch = StopwatchManager()
        for key, value in cfg.items():
            # Profile the module
            self.watch.initialize(key)

            # Add necessary geometry information
            value = deepcopy(value)
            if key != "recombination":
                value["num_tpcs"] = self.geo.tpc.num_chambers
            else:
                value["drift_dir"] = self.geo.tpc[0][0].drift_dir

            # Append
            self.modules[key] = calibrator_factory(key, value)

    def __call__(
        self,
        points: np.ndarray,
        values: np.ndarray,
        sources: Optional[np.ndarray] = None,
        run_id: Optional[int] = None,
        track: Optional[bool] = None,
        meta: Optional[Meta] = None,
        module_id: Optional[int] = None,
    ):
        """Main calibration driver.

        Parameters
        ----------
        points : np.ndarray, optional
            (N, 3) array of space point coordinates
        values : np.ndarray
            (N) array of depositions in ADC
        sources : np.ndarray, optional
            (N) array of [cryo, tpc] specifying which TPC produced each hit. If
            not specified, uses the closest TPC as calibration reference.
        run_id : int, optional
            ID of the run to get the calibration for. This is needed when using
            a database of corrections organized by run.
        track : bool, defaut `False`
            Whether the object is a track or not. If it is, the track gets
            segmented to evaluate local dE/dx and track angle.
        meta : Meta, optional
            If provided, use to convert the coordinates from image pixel
            coordinates to detector coordinates
        module_id : int, optional
            If provided, shift points to the requested module assuming that the
            points currently live in module ID 0

        Returns
        -------
        np.ndarray
            (N) array of calibrated depositions in ADC, e- or MeV
        """
        # If necessary, convert all points to detector coordinates
        if meta is not None:
            points = meta.to_cm(points, center=True)
        if module_id is not None:
            points = self.geo.translate(points, 0, module_id)

        # Create a mask for each of the TPC volume in the detector
        if sources is not None:
            tpc_indexes = []
            for module_id in range(self.geo.tpc.num_modules):
                for tpc_id in range(self.geo.tpc.num_chambers_per_module):
                    # Get the set of points associated with this TPC
                    tpc_index = self.geo.get_volume_index(sources, module_id, tpc_id)
                    tpc_indexes.append(tpc_index)

        else:
            assert (
                points is not None
            ), "If sources are not given, must provide points instead."
            tpc_indexes = self.geo.get_closest_tpc_indexes(points)

        # Loop over the TPCs, apply the relevant calibration corrections
        new_values = np.copy(values)
        for t in range(self.geo.tpc.num_chambers):
            # Restrict to the TPC of interest
            if len(tpc_indexes[t]) == 0:
                continue
            tpc_points = points[tpc_indexes[t]]
            tpc_values = values[tpc_indexes[t]]

            # Apply the transparency correction
            if "transparency" in self.modules:
                self.watch.start("transparency")
                tpc_values = self.modules["transparency"].process(
                    tpc_points, tpc_values, t, run_id
                )  # ADC
                self.watch.stop("transparency")

            # Apply the lifetime correction
            if "lifetime" in self.modules:
                self.watch.start("lifetime")
                tpc_values = self.modules["lifetime"].process(
                    tpc_points, tpc_values, self.geo, t, run_id
                )  # ADC
                self.watch.stop("lifetime")

            # Apply the gain correction
            if "gain" in self.modules:
                self.watch.start("gain")
                tpc_values = self.modules["gain"].process(tpc_values, t, run_id)  # e-
                self.watch.stop("gain")

            # Apply the recombination correction
            if "recombination" in self.modules:
                self.watch.start("recombination")
                tpc_values = self.modules["recombination"].process(
                    tpc_values, tpc_points, track
                )  # MeV
                self.watch.stop("recombination")

            # Append
            new_values[tpc_indexes[t]] = tpc_values

        return new_values
