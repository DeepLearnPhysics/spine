import numpy as np
import pytest

from spine.calib.function import CalibrationFunction


def test_calibration_function_evaluates_expression():
    function = CalibrationFunction("2.3 * exp(x) - 3")
    values = np.array([1.0, 2.0])

    assert np.allclose(function(values), 2.3 * np.exp(values) - 3)


def test_calibration_function_validates_expression_variable():
    with pytest.raises(ValueError, match="depend only"):
        CalibrationFunction("2 * y")


def test_calibration_function_validates_output_shape():
    function = CalibrationFunction("2 * x")
    function.function = lambda values: np.array(1.0)

    with pytest.raises(ValueError, match="same shape"):
        function(np.array([1.0, 2.0]))
