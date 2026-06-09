"""Loads all requested calibration modules and executes them
in the appropriate sequence."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from spine.geo import GeoManager
from spine.utils.factory import parse_module_config
from spine.utils.stopwatch import StopwatchManager

from .factories import calibrator_factory

if TYPE_CHECKING:  # pragma: no cover
    from spine.data import Meta


class CalibrationManager:
    """Manager in charge of applying all calibration-related corrections onto
    a set of 3D space points and their associated measured charge depositions.
    """

    def __init__(
        self, gain_applied: bool = False, **cfg: dict[str, Any] | None
    ) -> None:
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

        # Add the modules to a processor list in configuration order
        parsed = parse_module_config(cfg)
        names = [spec["name"] for spec in parsed.values()]

        # Make sure the essential calibration modules are present
        if not gain_applied and "recombination" in names and "gain" not in names:
            raise ValueError(
                "Must provide gain configuration if recombination is applied."
            )
        if not gain_applied and "gain" in names and "recombination" in names:
            if names.index("gain") > names.index("recombination"):
                raise ValueError(
                    "Gain calibration must be configured before recombination "
                    "calibration."
                )

        self.modules: dict[str, Any] = {}
        self.module_names: dict[str, str] = {}
        self.watch = StopwatchManager()
        for key, spec in parsed.items():
            # Profile the module
            self.watch.initialize(key)

            # Add necessary geometry information
            name = spec["name"]
            value = deepcopy(spec["cfg"])
            if name != "recombination":
                value["num_tpcs"] = self.geo.tpc.num_chambers
            else:
                value["drift_dir"] = self.geo.tpc[0][0].drift_dir

            # Append
            self.modules[key] = calibrator_factory(name, value)
            self.module_names[key] = name

    def __call__(
        self,
        points: NDArray[np.floating],
        values: NDArray[np.floating],
        sources: NDArray[np.integer] | None = None,
        run_id: int | None = None,
        track: bool | None = None,
        meta: Meta | None = None,
        module_id: int | None = None,
    ) -> NDArray[np.floating]:
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
            if points is None:
                raise ValueError(
                    "If sources are not given, must provide points instead."
                )
            tpc_indexes = self.geo.get_closest_tpc_indexes(points)

        # Loop over the TPCs, apply the relevant calibration corrections
        new_values = np.copy(values)
        for t in range(self.geo.tpc.num_chambers):
            # Restrict to the TPC of interest
            if len(tpc_indexes[t]) == 0:
                continue
            tpc_points = points[tpc_indexes[t]]
            tpc_values = values[tpc_indexes[t]]

            for key, module in self.modules.items():
                name = self.module_names[key]
                self.watch.start(key)
                if name == "transparency":
                    tpc_values = module.process(tpc_points, tpc_values, t, run_id)
                elif name == "lifetime":
                    tpc_values = module.process(
                        tpc_points, tpc_values, self.geo, t, run_id
                    )
                elif name == "gain":
                    tpc_values = module.process(tpc_values, t, run_id)
                elif name == "recombination":
                    tpc_values = module.process(tpc_values, tpc_points, track)
                else:
                    raise ValueError(f"Calibration module not recognized: {name}.")
                self.watch.stop(key)

            # Append
            new_values[tpc_indexes[t]] = tpc_values

        return new_values
