"""Applies conversion form ADC to number ionization electrons."""

from typing import List, Optional, Union

import numpy as np

from spine.utils.calib.database import CalibrationDatabase

__all__ = ["GainCalibrator"]


class GainCalibrator:
    """Converts all charge depositions in ADC to a number of electrons.

    It can either use a flat converstion factor or one per TPC in the detector. It
    can also use a SQLite database to provide gain values which depend on the run.
    """

    name = "gain"

    def __init__(
        self,
        num_tpcs: int,
        gain: Optional[Union[float, List[float]]] = None,
        gain_db: Optional[str] = None,
    ):
        """Initialize the recombination model and its constants.

        Parameters
        ----------
        num_tpcs : int
            Number of TPCs in the detector
        gain : Union[float, List[float]], optional
            Conversion factor from ADC to electrons (unique or per tpc)
        gain_db : str, optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets
        """
        # Must provide either a gain or a gain DB
        if (gain is None) == (gain_db is None):
            raise ValueError("Must provide either a gain or a gain_db.")

        # Initialize the gain values
        self.gain, self.gain_db = None, None
        if gain is not None:
            # If gain values are provided, make sure they are correct
            if isinstance(gain, (list, tuple, np.ndarray)) and len(gain) != num_tpcs:
                raise ValueError(
                    f"Gain must be a scalar or given per TPC ({num_tpcs})."
                )

            if isinstance(gain, (list, tuple, np.ndarray)):
                self.gain = gain
            else:
                self.gain = np.full(num_tpcs, gain)

        else:
            # If a database path is provided, load it
            assert gain_db is not None  # For the linter's sake
            self.gain_db = CalibrationDatabase(gain_db, num_tpcs)

    def process(
        self, values: np.ndarray, tpc_id: int, run_id: Optional[int] = None
    ) -> np.ndarray:
        """Converts deposition values from ADC to a number of electrons.

        Parameters
        ----------
        values : np.ndarray
            (N) array of depositions in ADC in a specific TPC
        tpc_id : int
            ID of the TPC to use

        Returns
        -------
        np.ndarray
            (N) array of depositions in number of electrons
        """
        # Fetch the gain value
        if self.gain_db is not None:
            assert run_id is not None, (
                "When using a gain database, the run_id must be provided "
                "to fetch the correct gain value."
            )
            gain_value = self.gain_db[run_id][tpc_id]

        elif self.gain is not None:
            gain_value = self.gain[tpc_id]

        else:
            raise RuntimeError("GainCalibrator not properly initialized.")

        # Apply the gain factor to all values in the current TPC
        return values * gain_value
