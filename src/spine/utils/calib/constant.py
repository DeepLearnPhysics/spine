"""Constants used in calibration modules."""

from typing import Dict, List, Optional, Union

import numpy as np

from .database import CalibrationDatabase


class CalibrationConstant:
    """Handles calibration constants which can be either scalar/per-TPC
    or provided via a database.
    """

    def __init__(
        self,
        num_tpcs: int,
        value: Optional[Union[float, List[float]]] = None,
        database: Optional[Union[str, Dict[int, Union[float, List[float]]]]] = None,
    ):
        """Initialize a calibration constant.

        Parameters
        ----------
        num_tpcs : int
            Number of TPCs in the detector
        value : Union[float, List[float]], optional
            Calibration constant (unique or per TPC)
        database : Union[str, Dict[int, Union[float, List[float]]]], optional
            Path to a SQLite db file or dictionary which maps run IDs to calibration
            constants or dictionary which maps run IDs to calibration constants
        """
        # Must provide either a value or a database
        if (value is None) == (database is None):
            raise ValueError("Must provide either a value or a database.")

        # Initialize the gain values
        self.value, self.database = None, None
        if value is not None:
            # If values are provided, make sure they are correct
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) != num_tpcs:
                raise ValueError(
                    "Calibration constant must be a scalar or given per TPC "
                    f"({num_tpcs}). Got {value} instead."
                )
            self.value = self.load_value(value, num_tpcs)

        else:
            # If a database path is provided, load it
            assert database is not None  # For the linter's sake
            if isinstance(database, dict):
                # If a dictionary is provided, make sure all entries are correct
                self.database = {
                    run_id: self.load_value(val, num_tpcs)
                    for run_id, val in database.items()
                }
            else:
                self.database = CalibrationDatabase(database, num_tpcs)

    def load_value(self, value: Union[float, List[float]], num_tpcs: int) -> np.ndarray:
        """Process value whether it is provided as a scalar or a list.

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
        if not isinstance(value, (list, tuple, np.ndarray)):
            return np.full(num_tpcs, value)

        assert len(value) == num_tpcs, (
            "Calibration constant must be specified as either a scalar "
            f"or a list of length {num_tpcs}."
        )
        return np.asarray(value)

    def get(self, tpc_id: int, run_id: Optional[int] = None) -> float:
        """Fetch the calibration constant for a specific TPC and run.

        Parameters
        ----------
        tpc_id : int
            ID of the TPC to use
        run_id : int, optional
            ID of the run to get the value for, if using a database

        Returns
        -------
        float
            Calibration constant for the specified TPC (and run ID)
        """
        # Fetch the calibration constant
        if self.database is not None:
            assert run_id is not None, (
                "When using a calibration database, the run_id must be provided "
                "to fetch the correct gain value."
            )
            return self.database[run_id][tpc_id]

        elif self.value is not None:
            return self.value[tpc_id]

        else:
            raise RuntimeError("CalibrationConstant not properly initialized.")
