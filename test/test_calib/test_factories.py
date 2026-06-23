import pytest

from spine.calib.factories import calibrator_factory
from spine.calib.gain import GainCalibrator


def test_calibrator_factory_instantiates_by_resolved_name():
    calibrator = calibrator_factory("gain", {"num_tpcs": 2, "gain": 2.0})

    assert isinstance(calibrator, GainCalibrator)


def test_calibrator_factory_does_not_mutate_config():
    cfg = {"num_tpcs": 2, "gain": 2.0}
    calibrator_factory("gain", cfg)

    assert cfg == {"num_tpcs": 2, "gain": 2.0}


def test_calibrator_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="Could not find"):
        calibrator_factory("unknown", {})
