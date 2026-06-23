import numpy as np
import pytest

import spine.calib.recombination as recombination_mod
from spine.calib.recombination import RecombinationCalibrator
from spine.constants import LAR_WION


def test_recombination_calibrator_applies_mip_correction():
    calibrator = RecombinationCalibrator(
        efield=0.5, drift_dir=np.array([1.0, 0.0, 0.0])
    )
    values = np.array([1000.0, 2000.0])

    corrected = calibrator.process(values, track=False)

    assert np.allclose(corrected, values * LAR_WION / calibrator.mip_recomb)


def test_recombination_calibrator_rejects_unknown_model():
    with pytest.raises(ValueError, match="not recognized"):
        RecombinationCalibrator(
            efield=0.5, drift_dir=np.array([1.0, 0.0, 0.0]), model="bad"
        )


def test_recombination_track_mode_requires_points():
    calibrator = RecombinationCalibrator(
        efield=0.5, drift_dir=np.array([1.0, 0.0, 0.0])
    )

    with pytest.raises(ValueError, match="without point coordinates"):
        calibrator.process(np.array([1.0]), track=True)


def test_recombination_birks_and_mbox_equations():
    birks = RecombinationCalibrator(
        efield=0.5, drift_dir=np.array([1.0, 0.0, 0.0]), model="birks"
    )
    mbox = RecombinationCalibrator(
        efield=0.5, drift_dir=np.array([1.0, 0.0, 0.0]), model="mbox_ell"
    )
    dedx = np.array([2.0, 4.0])
    dqdx = np.array([1000.0, 2000.0])

    assert np.allclose(birks.recombination_factor(dedx), birks.birks(dedx))
    assert np.allclose(birks.inv_recombination_factor(dqdx), birks.inv_birks(dqdx))
    assert np.allclose(mbox.recombination_factor(dedx, 0.5), mbox.mbox(dedx, 0.5))
    assert np.allclose(
        mbox.inv_recombination_factor(dqdx, 0.5), mbox.inv_mbox(dqdx, 0.5)
    )


def test_recombination_track_mode_uses_segment_dqdx(monkeypatch):
    def fake_segments(points, values, method, **kwargs):
        return (
            np.array([1000.0, 0.0]),
            None,
            None,
            [np.array([0, 1]), np.array([2])],
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            None,
        )

    monkeypatch.setattr(recombination_mod, "get_track_segment_dedxs", fake_segments)
    calibrator = RecombinationCalibrator(
        efield=0.5, drift_dir=np.array([1.0, 0.0, 0.0]), model="mbox_ell"
    )
    values = np.array([100.0, 200.0, 300.0])
    corrected = calibrator.process(values, np.zeros((3, 3)), track=True)
    expected_first = calibrator.inv_recombination_factor(1000.0, 1.0)
    expected_fallback = LAR_WION / calibrator.mip_recomb

    assert np.allclose(corrected[:2], values[:2] * expected_first)
    assert np.allclose(corrected[2], values[2] * expected_fallback)


def test_recombination_track_mode_without_angle_correction(monkeypatch):
    def fake_segments(points, values, method, **kwargs):
        return (
            np.array([1000.0]),
            None,
            None,
            [np.array([0, 1])],
            np.array([[1.0, 0.0, 0.0]]),
            None,
        )

    monkeypatch.setattr(recombination_mod, "get_track_segment_dedxs", fake_segments)
    calibrator = RecombinationCalibrator(
        efield=0.5, drift_dir=np.array([1.0, 0.0, 0.0]), model="mbox"
    )
    values = np.array([100.0, 200.0])
    corrected = calibrator.process(values, np.zeros((2, 3)), track=True)

    assert np.allclose(corrected, values * calibrator.inv_recombination_factor(1000.0))
