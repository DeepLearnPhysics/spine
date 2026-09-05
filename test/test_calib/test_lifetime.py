import numpy as np

from spine.calib.lifetime import LifetimeCalibrator


def test_lifetime_calibrator_applies_drift_correction(fake_geo):
    calibrator = LifetimeCalibrator(num_tpcs=2, lifetime=10.0, driftv=2.0)
    points = np.array([[4.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    values = np.array([1.0, 2.0])

    corrected = calibrator.process(points, values, fake_geo, tpc_id=0)

    assert np.allclose(corrected, values * np.exp(np.array([4.0, 10.0]) / 20.0))
    assert np.allclose(
        calibrator.unprocess(points, corrected, fake_geo, tpc_id=0), values
    )


def test_lifetime_calibrator_uses_database(fake_geo):
    calibrator = LifetimeCalibrator(
        num_tpcs=2,
        lifetime_db={100: [10.0, 20.0]},
        driftv_db={100: [2.0, 4.0]},
    )

    corrected = calibrator.process(
        np.array([[6.0, 0.0, 0.0]]),
        np.array([2.0]),
        fake_geo,
        tpc_id=1,
        run_id=100,
    )

    assert np.allclose(corrected, [2.0 * np.exp(4.0 / 80.0)])


def test_lifetime_calibrator_loads_scaled_per_tpc_columns(fake_geo, column_value_db):
    calibrator = LifetimeCalibrator(
        num_tpcs=2,
        lifetime_db=str(column_value_db),
        lifetime_db_value_keys=("east", "west"),
        lifetime_db_value_scale=1000.0,
        driftv=2.0,
    )

    corrected = calibrator.process(
        np.array([[6.0, 0.0, 0.0]]),
        np.array([2.0]),
        fake_geo,
        tpc_id=1,
        run_id=100,
    )

    assert np.allclose(corrected, [2.0 * np.exp(4.0 / 40.0)])
