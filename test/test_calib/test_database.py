import numpy as np
import pandas as pd
import pytest

from spine.calib.database import CalibrationDatabase, CalibrationLUT


def test_value_database_loads_active_iovs_and_queries_previous_run(value_db):
    db = CalibrationDatabase(str(value_db), num_tpcs=2)

    assert np.allclose(db[100], [2.0, 3.0])
    assert np.allclose(db[250], [4.0, 5.0])
    with pytest.raises(IndexError, match="No calibration information"):
        db[99]


def test_database_rejects_unknown_type(value_db):
    with pytest.raises(ValueError, match="Type of database"):
        CalibrationDatabase(str(value_db), num_tpcs=2, db_type="bad")


def test_value_database_requires_one_value_per_tpc(value_db):
    db = CalibrationDatabase(str(value_db), num_tpcs=2)
    bad_run = pd.DataFrame({"channel": [0], "gain": [1.0]})

    with pytest.raises(ValueError, match="one quantity"):
        db.load_values(bad_run, "gain")


def test_map_database_loads_luts(transparency_db):
    db = CalibrationDatabase(str(transparency_db), num_tpcs=4, db_type="map")

    maps = db[100]
    assert len(maps) == 4
    assert np.allclose(maps[0].query(np.array([[0.0, 0.25, 0.25]])), [1.0])
    assert np.allclose(maps[3].query(np.array([[0.0, 1.25, 1.25]])), [6.0])


def test_lut_clips_points_and_replaces_dummy_values():
    lut = CalibrationLUT(
        dims=[1, 2],
        bins=[2, 2],
        range=[[0.0, 2.0], [0.0, 2.0]],
        values=np.array([[1.0, -999.0], [3.0, 4.0]]),
    )

    points = np.array([[0.0, -1.0, 5.0], [0.0, 1.5, 1.5]])
    assert np.allclose(lut.query(points), [1.0, 4.0])
    assert [edge.tolist() for edge in lut.edges] == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]


def test_lut_validates_dimensions_and_values():
    with pytest.raises(ValueError, match="per dimension"):
        CalibrationLUT(
            dims=[1, 2],
            bins=[2],
            range=[[0.0, 2.0], [0.0, 2.0]],
            values=np.ones((2, 2)),
        )

    with pytest.raises(ValueError, match="one calibration value"):
        CalibrationLUT(
            dims=[1, 2],
            bins=[2, 2],
            range=[[0.0, 2.0], [0.0, 2.0]],
            values=np.ones((2, 3)),
        )
