"""Applies a nonlinear response function to deposition values."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .function import CalibrationFunction

__all__ = ["ResponseCalibrator"]


class ResponseCalibrator:
    """Transforms deposition values with a configurable response function."""

    name = "response"

    def __init__(self, response_func: str) -> None:
        """Initialize the response function.

        Parameters
        ----------
        response_func : str
            NumExpr expression which transforms the input values, represented
            by the variable ``x``.
        """
        self.response_func = CalibrationFunction(response_func)

    def process(self, values: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the response function to deposition values.

        Parameters
        ----------
        values : np.ndarray
            (N) array of deposition values

        Returns
        -------
        np.ndarray
            (N) array of transformed deposition values
        """
        return self.response_func(values)
