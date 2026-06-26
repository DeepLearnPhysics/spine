"""Applies conversion form ADC to number ionization electrons."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .constant import CalibrationConstant
from .function import CalibrationFunction

__all__ = ["GainCalibrator"]

CalibValue: TypeAlias = float | list[float] | tuple[float, ...] | NDArray[np.floating]
CalibDatabaseSource: TypeAlias = str | dict[int, CalibValue]


class GainCalibrator:
    """Converts all charge depositions in ADC to a number of electrons.

    It can either use a flat converstion factor or one per TPC in the detector. It
    can also use a SQLite database to provide gain values which depend on the run.
    """

    name = "gain"

    def __init__(
        self,
        num_tpcs: int,
        gain: CalibValue | None = None,
        gain_db: CalibDatabaseSource | None = None,
        gain_func: str | None = None,
    ) -> None:
        """Initialize the recombination model and its constants.

        Parameters
        ----------
        num_tpcs : int
            Number of TPCs in the detector
        gain : Union[float, List[float]], optional
            Conversion factor from ADC to electrons (unique or per tpc)
        gain_db : str, optional
            Path to a SQLite db file which maps [run, cryo, tpc] sets
        gain_func : str, optional
            NumExpr function to apply to the charge values directly
        """
        # Initialize the gain calibration constant or function.
        provided = [gain is not None, gain_db is not None, gain_func is not None]
        if sum(provided) != 1:
            raise ValueError("Must provide exactly one of gain, gain_db or gain_func.")

        self.gain: CalibrationConstant | None = None
        self.gain_func: CalibrationFunction | None = None
        if gain_func is not None:
            self.gain_func = CalibrationFunction(gain_func)
        else:
            self.gain = CalibrationConstant(
                num_tpcs=num_tpcs, value=gain, database=gain_db
            )

    def process(
        self, values: NDArray[np.floating], tpc_id: int, run_id: int | None = None
    ) -> NDArray[np.floating]:
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
        # Apply the gain function or factor to all values in the current TPC.
        if self.gain_func is not None:
            return self.gain_func(values)

        assert self.gain is not None
        return values * self.gain.get(tpc_id, run_id)
