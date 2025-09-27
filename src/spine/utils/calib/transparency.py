"""Apply wire transparency corrections."""

from warnings import warn

from .database import CalibrationDatabase

__all__ = ["TransparencyCalibrator"]


class TransparencyCalibrator:
    """Applies a correction on the amount of charge observed in a space point
    based on its position in the plane of the sensitive wires/pixels (yz).
    """

    name = "transparency"

    def __init__(self, transparency_db, num_tpcs, value_key="scale", run_id=None):
        """Load the calibration maps.

        Parameters
        ----------
        lifetime_db : str
            Path to a SQLite db file which maps [run, cryo, tpc] sets onto
            a specific transparency calibration map.
        num_tpcs : int
            Number of TPCs in the detector
        value_key: str, default 'scale'
            Database key which provides the calibration factor
        run_id : int
            Static run ID to use to fetch the transparency map
        """
        # Load the transparency database
        self.transparency = CalibrationDatabase(
            transparency_db, num_tpcs=num_tpcs, db_type="map", value_key=value_key
        )

        # Set a static run ID, if requested (for simulation)
        self.run_id = run_id
        if run_id is not None:
            warn(
                "The run ID provided by the event will be ignored in fetching "
                f"the calibration transparency map in favor of {run_id}."
            )

    def process(self, points, values, tpc_id, run_id):
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

        assert run_id is not None, "Must provide a run ID to get the transparency map."

        # Get the appropriate transparency map for this run
        transparency_lut = self.transparency[run_id]

        # Get the transparency correction for each position in the image
        return values / transparency_lut[tpc_id].query(points)
