"""Applies conversion form ADC to number ionization electrons."""

from typing import Dict, List, Optional, Union

import numpy as np

from .constant import CalibrationConstant

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
        gain_db: Optional[Union[str, Dict[int, Union[float, List[float]]]]] = None,
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
        # Initialize the gain calibration constant
        self.gain = CalibrationConstant(num_tpcs=num_tpcs, value=gain, database=gain_db)

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
        # Apply the gain factor to all values in the current TPC
        return values * self.gain.get(tpc_id, run_id)
