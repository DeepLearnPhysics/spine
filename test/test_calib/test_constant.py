import numpy as np
import pytest

from spine.calib.constant import CalibrationConstant


def test_constant_accepts_scalar_and_per_tpc_values():
    scalar = CalibrationConstant(num_tpcs=2, value=3.0)
    per_tpc = CalibrationConstant(num_tpcs=2, value=[2.0, 4.0])

    assert np.allclose(scalar.value, [3.0, 3.0])
    assert scalar.get(1) == 3.0
    assert np.allclose(per_tpc.value, [2.0, 4.0])
    assert per_tpc.get(1) == 4.0


def test_constant_accepts_dict_database():
    constant = CalibrationConstant(num_tpcs=2, database={100: [2.0, 3.0]})

    assert constant.get(0, run_id=100) == 2.0
    assert constant.get(1, run_id=100) == 3.0


def test_constant_validates_source_and_shape():
    with pytest.raises(ValueError, match="either a value or a database"):
        CalibrationConstant(num_tpcs=2)

    with pytest.raises(ValueError, match="either a value or a database"):
        CalibrationConstant(num_tpcs=2, value=1.0, database={})

    with pytest.raises(ValueError, match="per TPC"):
        CalibrationConstant(num_tpcs=2, value=[1.0, 2.0, 3.0])

    scalar = CalibrationConstant(num_tpcs=2, value=1.0)
    with pytest.raises(ValueError, match="list of length"):
        scalar.load_value([1.0], num_tpcs=2)

    constant = CalibrationConstant(num_tpcs=2, database={100: [1.0, 2.0]})
    with pytest.raises(ValueError, match="run_id"):
        constant.get(0)

    constant.database = None
    with pytest.raises(RuntimeError, match="not properly initialized"):
        constant.get(0)
