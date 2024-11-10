"""Loads all requested calibration modules and executes them
in the appropriate sequence."""

import numpy as np

from spine.utils.geo import Geometry
from spine.utils.stopwatch import StopwatchManager

from .factories import calibrator_factory


class CalibrationManager:
    """Manager in charge of applying all calibration-related corrections onto
    a set of 3D space points and their associated measured charge depositions.
    """

    def __init__(self, geometry, **cfg):
        """Initialize the manager.

        Parameters
        ----------
        geometry : dict
            Geometry configuration
        **cfg : dict, optional
            Calibrator configurations
        """
        # Initialize the geometry model shared across all modules
        self.geo = Geometry(**geometry)

        # Make sure the essential calibration modules are present
        assert 'recombination' not in cfg or 'gain' in cfg, (
                "Must provide gain configuration if recombination is applied.")

        # Add the modules to a processor list in decreasing order of priority
        self.modules = {}
        self.watch   = StopwatchManager()
        for key, value in cfg.items():
            # Profile the module
            self.watch.initialize(key)

            # Add necessary geometry information
            if key != 'recombination':
                value['num_tpcs'] = self.geo.tpc.num_tpcs
            else:
                value['drift_dir'] = self.geo.tpc[0, 0].drift_dir

            # Append
            self.modules[key] = calibrator_factory(key, value)

    def __call__(self, points, values, sources=None, run_id=None,
                 dedx=None, track=None):
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
        dedx : float, optional
            If specified, use a flat value of dE/dx in MeV/cm to apply
            the recombination correction.
        track : bool, defaut `False`
            Whether the object is a track or not. If it is, the track gets
            segmented to evaluate local dE/dx and track angle.

        Returns
        -------
        np.ndarray
            (N) array of calibrated depositions in ADC, e- or MeV
        """
        # Create a mask for each of the TPC volume in the detector
        if sources is not None:
            tpc_indexes = []
            for t in range(self.geo.num_tpcs):
                # Get the set of points associated with this TPC
                module_id = t//self.geo.num_tpcs_per_module
                tpc_id = t%self.geo.num_tpcs_per_module
                tpc_index = self.geo.get_tpc_index(sources, module_id, tpc_id)
                tpc_indexes.append(tpc_index)
        else:
            assert points is not None, (
                    "If sources are not given, must provide points instead.")
            tpc_indexes = self.geo.get_closest_tpc_indexes(points)

        # Loop over the TPCs, apply the relevant calibration corrections
        new_values = np.copy(values)
        for t in range(self.geo.num_tpcs):
            # Restrict to the TPC of interest
            if len(tpc_indexes[t]) == 0:
                continue
            tpc_points = points[tpc_indexes[t]]
            tpc_values = values[tpc_indexes[t]]

            # Apply the transparency correction
            if 'transparency' in self.modules:
                assert run_id is not None, (
                        "Must provide a run ID to get the transparency map.")
                self.watch.start('transparency')
                tpc_values = self.modules['transparency'].process(
                        tpc_points, tpc_values, t, run_id) # ADC
                self.watch.stop('transparency')

            # Apply the lifetime correction
            if 'lifetime' in self.modules:
                self.watch.start('lifetime')
                tpc_values = self.modules['lifetime'].process(
                        tpc_points, tpc_values, self.geo, t, run_id) # ADC
                self.watch.stop('lifetime')

            # Apply the gain correction
            if 'gain' in self.modules:
                self.watch.start('gain')
                tpc_values = self.modules['gain'].process(tpc_values, t) # e-
                self.watch.stop('gain')

            # Apply the recombination
            if 'recombination' in self.modules:
                self.watch.start('recombination')
                tpc_values = self.modules['recombination'].process(
                        tpc_values, tpc_points, dedx, track) # MeV
                self.watch.stop('recombination')

            # Append
            new_values[tpc_indexes[t]] = tpc_values

        return new_values
