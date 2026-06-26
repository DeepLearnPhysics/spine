"""Functions used in calibration modules."""

from __future__ import annotations

import numexpr as ne
import numpy as np
from numpy.typing import NDArray


class CalibrationFunction:
    """Handles calibration functions evaluated on charge arrays."""

    def __init__(self, expression: str, variable: str = "x") -> None:
        """Initialize a calibration function.

        Parameters
        ----------
        expression : str
            NumExpr expression to evaluate.
        variable : str, default "x"
            Name of the input array in the expression.
        """
        self.expression = expression
        self.variable = variable
        self.function = ne.NumExpr(expression)

        if self.function.input_names != (variable,):
            raise ValueError(
                "Calibration function expression must depend only on the "
                f"`{variable}` variable. Got inputs {self.function.input_names}."
            )

    def __call__(self, values: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate the calibration function on an array of values.

        Parameters
        ----------
        values : np.ndarray
            Input values to transform.

        Returns
        -------
        np.ndarray
            Transformed values.
        """
        result = self.function(values)
        result = np.asarray(result)

        if result.shape != values.shape:
            raise ValueError(
                "Calibration function must return an array with the same shape "
                f"as the input. Got {result.shape}, expected {values.shape}."
            )

        return result
