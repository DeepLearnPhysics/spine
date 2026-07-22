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
