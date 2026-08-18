"""Apply wire transparency corrections."""

from __future__ import annotations

from typing import cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from spine.utils.conditional import ROOT, ROOT_AVAILABLE

from .database import CalibrationDatabase, CalibrationLUT

__all__ = ["TransparencyCalibrator"]


class TransparencyCalibrator:
    """Applies a correction on the amount of charge observed in a space point
    based on its position in the plane of the sensitive wires/pixels (yz).
    """

    name = "transparency"
    _map_types = ("correction", "deviation")

    def __init__(
        self,
        transparency_db: str | None = None,
        num_tpcs: int | None = None,
        value_key: str = "scale",
        run_id: int | None = None,
        transparency_file: str | None = None,
        map_pattern: str = "CzyHist_{plane_id}_{tpc_id}",
        plane_id: int = 2,
        map_type: str | None = None,
    ) -> None:
        """Load the calibration maps.

        Parameters
        ----------
        transparency_db : str, optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific transparency calibration map.
        num_tpcs : int
            Number of TPCs in the detector
        value_key: str, default 'scale'
            Database key which provides the calibration factor
        run_id : int
            Static run ID to use to fetch the transparency map
        transparency_file : str, optional
            ROOT file containing one static transparency map per TPC. Mutually
            exclusive with ``transparency_db``.
        map_pattern : str, default 'CzyHist_{plane_id}_{tpc_id}'
            Format pattern used to identify ROOT histograms.
        plane_id : int, default 2
            Wire plane substituted into ``map_pattern``.
        map_type : {'correction', 'deviation'}, optional
            Meaning of the stored map values. Corrections multiply the input;
            deviations are divided out. Defaults to ``'deviation'`` for a
            database and ``'correction'`` for a static ROOT file.
        """
        # Make sure the detector size and exactly one map source are provided
        if num_tpcs is None:
            raise ValueError("Must provide the number of TPCs.")
        if (transparency_db is None) == (transparency_file is None):
            raise ValueError(
                "Must provide exactly one of transparency_db or transparency_file."
            )

        # Resolve and validate how the stored map values should be applied
        if map_type is None:
            map_type = "deviation" if transparency_db is not None else "correction"
        if map_type not in self._map_types:
            raise ValueError(
                f"Transparency map type not recognized: {map_type}. "
                f"Must be one of {self._map_types}."
            )
        self.map_type = map_type

        # Initialize the possible map sources
        self.database: CalibrationDatabase | None = None
        self.maps: list[CalibrationLUT] | None = None
        self.transparency: CalibrationDatabase | list[CalibrationLUT]

        # Load the requested run-dependent database or static map collection
        if transparency_db is not None:
            self.database = CalibrationDatabase(
                transparency_db,
                num_tpcs=num_tpcs,
                db_type="map",
                value_key=value_key,
            )
            self.transparency = self.database
        else:
            assert transparency_file is not None
            self.maps = self._load_root_maps(
                transparency_file, num_tpcs, map_pattern, plane_id
            )
            self.transparency = self.maps

        # Set a static run ID, if requested (for simulation)
        self.run_id = run_id
        if run_id is not None and self.database is not None:
            warn(
                "The run ID provided by the event will be ignored in fetching "
                f"the calibration transparency map in favor of {run_id}."
            )

    @staticmethod
    def _load_root_maps(
        transparency_file: str,
        num_tpcs: int,
        map_pattern: str,
        plane_id: int,
    ) -> list[CalibrationLUT]:
        """Load static per-TPC transparency maps from a ROOT file.

        Parameters
        ----------
        transparency_file : str
            Path to the ROOT file containing the maps.
        num_tpcs : int
            Number of TPC maps to load.
        map_pattern : str
            Histogram-name pattern formatted with ``plane_id`` and ``tpc_id``.
        plane_id : int
            Wire-plane identifier substituted into ``map_pattern``.

        Returns
        -------
        list[CalibrationLUT]
            Maps ordered by TPC ID.

        Raises
        ------
        ImportError
            If ROOT is unavailable.
        OSError
            If the ROOT file cannot be opened.
        KeyError
            If a requested per-TPC histogram is missing.
        """
        # Make sure the optional ROOT dependency is available
        if not ROOT_AVAILABLE:
            raise ImportError("ROOT is required to load transparency ROOT files.")

        # Open the map file and validate its state
        root_file = ROOT.TFile.Open(  # pylint: disable=E1101
            str(transparency_file), "r"
        )
        if not root_file or root_file.IsZombie():
            raise OSError(f"Could not open transparency ROOT file: {transparency_file}")

        try:
            # Load one histogram-backed look-up table per TPC
            maps = []
            for tpc_id in range(num_tpcs):
                # Build the name of the histogram associated with this TPC
                hist_name = map_pattern.format(
                    plane_id=plane_id,
                    tpc_id=tpc_id,
                )

                # Fetch the requested histogram and make sure it exists
                histogram = root_file.Get(hist_name)
                if not histogram:
                    raise KeyError(
                        f"Could not find histogram '{hist_name}' in "
                        f"{transparency_file}."
                    )

                # Convert the ROOT histogram to the common LUT representation
                maps.append(
                    CalibrationLUT.from_root_histogram(
                        histogram,
                        axis_dims=(2, 1),
                    )
                )

        finally:
            # ROOT files must be closed even when a requested map is missing
            root_file.Close()

        return maps

    def process(
        self,
        points: NDArray[np.floating],
        values: NDArray[np.floating],
        tpc_id: int,
        run_id: int | None,
    ) -> NDArray[np.floating]:
        """Apply the transparency correction.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) array of point coordinates
        values : np.ndarray
            (N) array of values associated with each point
        tpc_id : int
            ID of the TPC to use
        run_id : int
            Used to get the appropriate transparency map

        Returns
        -------
        np.ndarray
            (N) array of corrected values
        """
        # If a static run ID was provided in the configuration, override
        if self.run_id is not None:
            run_id = self.run_id

        if self.database is not None and run_id is None:
            raise ValueError("Must provide a run ID to get the transparency map.")

        # Get the appropriate transparency map for this run
        if self.database is not None:
            assert run_id is not None
            transparency_luts = cast(list[CalibrationLUT], self.database[run_id])
        else:
            assert self.maps is not None
            transparency_luts = self.maps

        # Apply the map according to the meaning of its stored values
        factors = transparency_luts[tpc_id].query(points)
        if self.map_type == "correction":
            return values * factors

        return values / factors
