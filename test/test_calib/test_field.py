import pytest

from spine.calib.field import FieldCalibrator


def test_field_calibrator_is_unimplemented():
    with pytest.raises(NotImplementedError, match="not yet available"):
        FieldCalibrator()
