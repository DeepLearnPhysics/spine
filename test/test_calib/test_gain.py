import numpy as np
import pytest

from spine.calib.gain import GainCalibrator


def test_gain_calibrator_applies_scalar_and_per_tpc_gain():
    values = np.array([1.0, 2.0])

    scalar = GainCalibrator(num_tpcs=2, gain=2.0)
    per_tpc = GainCalibrator(num_tpcs=2, gain=[2.0, 3.0])

    assert np.allclose(scalar.process(values, tpc_id=1), [2.0, 4.0])
    assert np.allclose(per_tpc.process(values, tpc_id=1), [3.0, 6.0])


def test_gain_calibrator_uses_database(value_db):
    calibrator = GainCalibrator(num_tpcs=2, gain_db=str(value_db))

    assert np.allclose(
        calibrator.process(np.array([2.0]), tpc_id=1, run_id=250), [10.0]
    )


def test_gain_calibrator_applies_gain_function():
    calibrator = GainCalibrator(num_tpcs=2, gain_func="2.3 * exp(x) - 3")
    values = np.array([1.0, 2.0])

    assert np.allclose(calibrator.process(values, tpc_id=1), 2.3 * np.exp(values) - 3)


def test_gain_calibrator_validates_gain_source():
    with pytest.raises(ValueError, match="exactly one"):
        GainCalibrator(num_tpcs=2)

    with pytest.raises(ValueError, match="exactly one"):
        GainCalibrator(num_tpcs=2, gain=2.0, gain_func="2 * x")

    with pytest.raises(ValueError, match="depend only"):
        GainCalibrator(num_tpcs=2, gain_func="2 * y")
