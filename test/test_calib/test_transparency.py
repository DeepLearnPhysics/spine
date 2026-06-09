import numpy as np
import pytest

from spine.calib.transparency import TransparencyCalibrator


def test_transparency_calibrator_applies_map_correction(transparency_db):
    calibrator = TransparencyCalibrator(str(transparency_db), num_tpcs=4)

    corrected = calibrator.process(
        np.array([[0.0, 1.25, 1.25]]),
        np.array([12.0]),
        tpc_id=3,
        run_id=100,
    )

    assert np.allclose(corrected, [2.0])


def test_transparency_calibrator_static_run_overrides_event_run(transparency_db):
    calibrator = TransparencyCalibrator(str(transparency_db), num_tpcs=4, run_id=100)

    corrected = calibrator.process(
        np.array([[0.0, 0.25, 0.25]]),
        np.array([8.0]),
        tpc_id=1,
        run_id=None,
    )

    assert np.allclose(corrected, [4.0])


def test_transparency_calibrator_requires_run_id(transparency_db):
    calibrator = TransparencyCalibrator(str(transparency_db), num_tpcs=4)

    with pytest.raises(ValueError, match="Must provide a run ID"):
        calibrator.process(np.zeros((1, 3)), np.ones(1), tpc_id=0, run_id=None)
