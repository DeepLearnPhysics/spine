"""Apply electron lifetime corrections."""

from typing import Dict, List, Optional, Union

import numpy as np

from spine.geo.base import Geometry

from .constant import CalibrationConstant

__all__ = ["LifetimeCalibrator"]


class LifetimeCalibrator:
    """Applies a correction based on drift electron lifetime and the distance
    from the ionization point to the closest readout plane.
    """

    name = "lifetime"

    def __init__(
        self,
        num_tpcs: int,
        lifetime: Optional[Union[float, List[float]]] = None,
        driftv: Optional[Union[float, List[float]]] = None,
        lifetime_db: Optional[Union[str, Dict[int, Union[float, List[float]]]]] = None,
        driftv_db: Optional[Union[str, Dict[int, Union[float, List[float]]]]] = None,
    ):
        """Load the information needed to make a lifetime correction.

        Parameters
        ----------
        num_tpcs : int
            Number of TPCs in the detector
        lifetime : Union[float, List[float]], optional
            Specifies the electron lifetime in microseconds. If `List`, it
            should map a tpc ID onto a specific value.
        driftv : Union[float, List[float]], optional
            Specifies the electron drift velocity in cm/us. If `List`, it
            should map a tpc ID onto a specific value.
        lifetime_db : Union[str, Dict[int, Union[float, List[float]]]], optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific lifetime value in microseconds, or a dictionary which
            maps run IDs to lifetime values.
        driftv_db : Union[str, Dict[int, Union[float, List[float]]]], optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific electron drift velocity value in cm/us, or a dictionary which
            maps run IDs to drift velocity values.
        """
        # Initialize lifetime and drift velocity calibration constants
        self.lifetime = CalibrationConstant(
            num_tpcs=num_tpcs, value=lifetime, database=lifetime_db
        )
        self.driftv = CalibrationConstant(
            num_tpcs=num_tpcs, value=driftv, database=driftv_db
        )

    def process(
        self,
        points: np.ndarray,
        values: np.ndarray,
        geo: Geometry,
        tpc_id: int,
        run_id: Optional[int] = None,
    ):
        """Apply the lifetime correction.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) array of point coordinates
        values : np.ndarray
            (N) array of values associated with each point
        geo : Geometry
            Detector geometry object
        tpc_id : int
            ID of the TPC to use
        run_id : int, optional
            If provided, used to get the appropriate lifetime/drift velocities

        Returns
        -------
        np.ndarray
            (N) array of corrected values
        """
        # Get the corrections lifetimes/drift velocities
        lifetime = self.lifetime.get(tpc_id, run_id)
        driftv = self.driftv.get(tpc_id, run_id)

        # Compute the distance to the anode plane
        mod = tpc_id // geo.tpc.num_chambers_per_module
        tpc = tpc_id % geo.tpc.num_chambers_per_module
        daxis = geo.tpc[mod][tpc].drift_axis
        position = geo.tpc[mod][tpc].anode_pos
        drifts = np.abs(points[:, daxis] - position)

        # Clip down to the physical range of possible drift distances
        max_drift = geo.tpc[mod][tpc].dimensions[daxis]
        drifts = np.clip(drifts, 0.0, max_drift)

        # Convert the drift distances to correction factors
        corrections = np.exp(drifts / lifetime / driftv)
        return corrections * values
