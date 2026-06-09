import numpy as np

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
