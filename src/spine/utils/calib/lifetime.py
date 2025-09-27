"""Apply electron lifetime corrections."""

import numpy as np

from .database import CalibrationDatabase

__all__ = ["LifetimeCalibrator"]


class LifetimeCalibrator:
    """Applies a correction based on drift electron lifetime and the distance
    from the ionization point to the closest readout plane.
    """

    name = "lifetime"

    def __init__(
        self, num_tpcs, lifetime=None, driftv=None, lifetime_db=None, driftv_db=None
    ):
        """Load the information needed to make a lifetime correction.

        Parameters
        ----------
        num_tpcs : int
            Number of TPCs in the detector
        lifetime : Union[float, list, dict], optional
            Specifies the electron lifetime in microseconds. If `list`, it
            should map a tpc ID onto a specific value. If `dict`, it should
            map each run ID to either a float or a list as defined above.
        driftv : Union[float, list, dict], optional
            Specifies the electron drift velocity in cm/us. If `dict`, it
            should map a tpc ID onto a specific value. If `dict`, it should
            map each run ID to either a float or a list as defined above.
        lifetime_db : str, optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific lifetime value in microseconds.
        driftv_db : str, optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific electron drift velocity value in cm/us.
        """
        # Loop over the two quantities needed for this module, initialize
        sources = {"lifetime": (lifetime, lifetime_db), "driftv": (driftv, driftv_db)}
        for quant, (source, source_db) in sources.items():
            # Make sure that the quantity is provided either directly or DB
            assert (source is None) ^ (source_db is None), (
                f"Must provide either the {quant} directly or point to "
                f"a SQLite database through {quant}_db, not both."
            )

            # Set whether the quantity is provided per run or not
            setattr(
                self,
                f"{quant}_per_run",
                source_db is not None or isinstance(source, dict),
            )

            # Dispatch
            if source is not None:
                # If static values are specified, store them
                if not getattr(self, f"{quant}_per_run"):
                    setattr(self, quant, self.get_quantity(source, num_tpcs))
                else:
                    full_dict = {}
                    for key, value in source.items():
                        full_dict[key] = self.get_quantity(value, num_tpcs)
                    setattr(self, quant, full_dict)

            else:
                # If database path is provided, load it
                setattr(self, quant, CalibrationDatabase(source_db, num_tpcs))

    @staticmethod
    def get_quantity(value, num_tpcs):
        """Process quantity weather it is provided as a scalar or a list.

        Parameters
        ----------
        value : Union[float, list]
            Specifies the quantity value as a scalar or a list
        num_tpcs : int
            Number of TPCs in the detector

        Returns
        -------
        np.ndarray
             List of quantities, one per TPC
        """
        # Inititalize quantity
        if np.isscalar(value):
            return np.full(num_tpcs, value)

        assert len(value) == num_tpcs, (
            "`lifetime` and `driftv' must be specified as either scalars "
            f"or as lists with one value per TPC ({num_tpcs})."
        )

        return value

    def process(self, points, values, geo, tpc_id, run_id=None):
        """
        Apply the lifetime correction.

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
        lifetime = self.lifetime
        driftv = self.driftv
        if self.lifetime_per_run or self.driftv_per_run:
            assert run_id is not None, (
                "When `lifetime` or `driftv` is defined per run ID, "
                "must provide a run ID to the process function."
            )
            if self.lifetime_per_run:
                lifetime = self.lifetime[run_id]
            if self.driftv_per_run:
                driftv = self.driftv[run_id]

        # Compute the distance to the anode plane
        m = tpc_id // geo.tpc.num_chambers_per_module
        t = tpc_id % geo.tpc.num_chambers_per_module
        daxis = geo.tpc[m, t].drift_axis
        position = geo.tpc[m, t].anode_pos
        drifts = np.abs(points[:, daxis] - position)

        # Clip down to the physical range of possible drift distances
        max_drift = geo.tpc[m, t].dimensions[daxis]
        drifts = np.clip(drifts, 0.0, max_drift)

        # Convert the drift distances to correction factors
        corrections = np.exp(drifts / lifetime[tpc_id] / driftv[tpc_id])
        return corrections * values
