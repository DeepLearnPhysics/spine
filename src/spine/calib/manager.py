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
        self.update_points = "field" in names

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
            if name == "recombination":
                value["drift_dir"] = self.geo.tpc[0][0].drift_dir
            elif name != "response":
                value["num_tpcs"] = self.geo.tpc.num_chambers

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
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
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
            (N, 3) array of calibrated point coordinates
        np.ndarray
            (N) array of calibrated depositions in ADC, e- or MeV
        """
        # If necessary, convert all points to detector coordinates
        orig_points = points
        if meta is not None:
            points = meta.to_cm(points, center=True)
        if module_id is not None:
            points = self.geo.translate(points, 0, module_id)

        # Create a mask for each TPC volume in the detector
        tpc_indexes = self._get_tpc_indexes(points, sources)

        # Loop over the TPCs, apply the relevant calibration corrections
        new_points = np.copy(points) if self.update_points else orig_points
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
                if name == "field":
                    tpc_points = module.process(tpc_points, t)
                elif name == "transparency":
                    tpc_values = module.process(tpc_points, tpc_values, t, run_id)
                elif name == "lifetime":
                    tpc_values = module.process(
                        tpc_points, tpc_values, self.geo, t, run_id
                    )
                elif name == "gain":
                    tpc_values = module.process(tpc_values, t, run_id)
                elif name == "response":
                    tpc_values = module.process(tpc_values)
                elif name == "recombination":
                    tpc_values = module.process(tpc_values, tpc_points, track)
                else:
                    raise ValueError(f"Calibration module not recognized: {name}.")
                self.watch.stop(key)

            # Append
            if self.update_points:
                new_points[tpc_indexes[t]] = tpc_points
            new_values[tpc_indexes[t]] = tpc_values

        if self.update_points:
            if module_id is not None:
                new_points = self.geo.translate(new_points, module_id, 0)
            if meta is not None:
                new_points = meta.to_px(new_points, floor=True)

        return new_points, new_values

    def process_points(
        self,
        points: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Apply coordinate-changing calibration modules to arbitrary points.

        This is intended for reconstructed position attributes such as particle
        start/end points. It only runs modules which alter coordinates
        (currently the field calibrator); charge-only calibrations are skipped.
        Input points must be expressed in detector coordinates. Their geometric
        location determines the applicable field map; source provenance is not
        needed.

        Parameters
        ----------
        points : np.ndarray
            ``(N, 3)`` point coordinates in detector units.

        Returns
        -------
        np.ndarray
            Corrected point coordinates with the same dtype as ``points``.
        """
        if not self.update_points or len(points) == 0:
            return points

        tpc_indexes = self._get_tpc_indexes(points, None)
        new_points = np.copy(points)
        for t, tpc_index in enumerate(tpc_indexes):
            if len(tpc_index) == 0:
                continue

            tpc_points = points[tpc_index]
            for key, module in self.modules.items():
                if self.module_names[key] != "field":
                    continue

                self.watch.start(key)
                tpc_points = module.process(tpc_points, t)
                self.watch.stop(key)

            new_points[tpc_index] = tpc_points

        return new_points

    def _get_tpc_indexes(
        self,
        points: NDArray[np.floating],
        sources: NDArray[np.integer] | None,
    ) -> list[NDArray[np.integer]]:
        """Build one point index for each TPC in the detector."""
        if sources is None:
            if points is None:
                raise ValueError(
                    "If sources are not given, must provide points instead."
                )
            return self.geo.get_closest_tpc_indexes(points)

        tpc_indexes = []
        for source_module_id in range(self.geo.tpc.num_modules):
            for tpc_id in range(self.geo.tpc.num_chambers_per_module):
                tpc_index = self.geo.get_volume_index(sources, source_module_id, tpc_id)
                tpc_indexes.append(tpc_index)

        return tpc_indexes
