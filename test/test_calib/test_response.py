import numpy as np
import pytest

from spine.calib.response import ResponseCalibrator


def test_response_calibrator_applies_response_function():
    calibrator = ResponseCalibrator(response_func="2.3 * exp(x) - 3")
    values = np.array([1.0, 2.0])

    assert np.allclose(calibrator.process(values), 2.3 * np.exp(values) - 3)


def test_response_calibrator_validates_response_function():
    with pytest.raises(ValueError, match="depend only"):
        ResponseCalibrator(response_func="2 * y")


def test_response_calibrator_requires_explicit_inverse():
    """Response functions should only be inverted when explicitly defined."""
    values = np.array([1.0, 2.0])
    calibrator = ResponseCalibrator("2 * x", "x / 2")

    assert np.allclose(calibrator.unprocess(calibrator.process(values)), values)
    with pytest.raises(ValueError, match="inverse_response_func"):
        ResponseCalibrator("2 * x").unprocess(values)
