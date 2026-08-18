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


def test_value_database_loads_per_tpc_columns_with_scale(column_value_db):
    db = CalibrationDatabase(
        str(column_value_db),
        num_tpcs=2,
        value_keys=("east", "west"),
        value_scale=1000.0,
    )

    assert np.allclose(db[100], [10.0, 20.0])
    assert np.allclose(db[250], [30.0, 40.0])


def test_value_database_validates_column_mapping(value_db):
    with pytest.raises(ValueError, match="one database value key per TPC"):
        CalibrationDatabase(str(value_db), num_tpcs=2, value_keys=("gain",))

    db = CalibrationDatabase(str(value_db), num_tpcs=2)
    rows = pd.DataFrame({"east": [1.0, 2.0], "west": [3.0, 4.0]})
    with pytest.raises(ValueError, match="one row per IOV"):
        db.load_values(rows, "", value_keys=("east", "west"))


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
        ranges=[[0.0, 2.0], [0.0, 2.0]],
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
            ranges=[[0.0, 2.0], [0.0, 2.0]],
            values=np.ones((2, 2)),
        )

    with pytest.raises(ValueError, match="one calibration value"):
        CalibrationLUT(
            dims=[1, 2],
            bins=[2, 2],
            ranges=[[0.0, 2.0], [0.0, 2.0]],
            values=np.ones((2, 3)),
        )


def test_lut_builds_from_dataframe_independent_of_row_order():
    dataframe = pd.DataFrame(
        {
            "ybin": [1, 0, 1, 0],
            "zbin": [1, 1, 0, 0],
            "ylow": [1.0, 0.0, 1.0, 0.0],
            "yhigh": [2.0, 1.0, 2.0, 1.0],
            "zlow": [1.0, 1.0, 0.0, 0.0],
            "zhigh": [2.0, 2.0, 1.0, 1.0],
            "scale": [4.0, 2.0, 3.0, 1.0],
        }
    )

    lut = CalibrationLUT.from_dataframe(dataframe, "scale")

    assert np.allclose(lut.values, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    ("dataframe", "match"),
    [
        (pd.DataFrame(), "empty table"),
        (pd.DataFrame({"ybin": [-1], "zbin": [0]}), "non-negative"),
        (
            pd.DataFrame({"ybin": [0, 1], "zbin": [0, 1]}),
            "exactly one calibration value per bin",
        ),
    ],
)
def test_lut_rejects_invalid_dataframes(dataframe, match):
    with pytest.raises(ValueError, match=match):
        CalibrationLUT.from_dataframe(dataframe, "scale")


def test_lut_builds_from_root_histogram_and_transposes_axes():
    histogram = FakeTH2([[2.0, 0.0], [4.0, np.nan]])

    lut = CalibrationLUT.from_root_histogram(histogram, reciprocal=True)

    assert lut.dims == [1, 2]
    assert np.all(lut.bins == [2, 2])
    assert np.allclose(lut.range, [[-1.0, 1.0], [10.0, 14.0]])
    assert np.allclose(lut.values, [[0.5, 0.25], [1.0, 1.0]])


def test_lut_root_histogram_replaces_missing_values_without_reciprocal():
    histogram = FakeTH2([[2.0, 0.0], [4.0, np.nan]])

    lut = CalibrationLUT.from_root_histogram(histogram)

    assert np.allclose(lut.values, [[2.0, 4.0], [1.0, 1.0]])


def test_lut_root_histogram_requires_distinct_axis_dimensions():
    with pytest.raises(ValueError, match="two distinct"):
        CalibrationLUT.from_root_histogram(FakeTH2([[1.0]]), axis_dims=(1, 1))


class FakeAxis:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def GetXmin(self):
        return self.low

    def GetXmax(self):
        return self.high


class FakeTH2:
    def __init__(self, values):
        self.values = np.asarray(values)

    def GetXaxis(self):
        return FakeAxis(10.0, 14.0)

    def GetYaxis(self):
        return FakeAxis(-1.0, 1.0)

    def GetNbinsX(self):
        return self.values.shape[0]

    def GetNbinsY(self):
        return self.values.shape[1]

    def GetBinContent(self, ix, iy):
        return self.values[ix - 1, iy - 1]
