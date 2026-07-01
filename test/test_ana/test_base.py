from __future__ import annotations

import pytest

import spine.ana.base as base_mod
from spine.ana.base import AnaBase


class DummyRunInfo:
    def scalar_dict(self):
        return {"run": 1}


class DummyAna(AnaBase):
    name = "dummy"

    def process(self, data):
        self.append("out", value=data["value"])
        return {"updated": data["value"] + 1}


class DummyWriter:
    def __init__(self, file_name, append=False, overwrite=False, buffer_size=1):
        self.file_name = file_name
        self.append_file = append
        self.overwrite_file = overwrite
        self.buffer_size = buffer_size
        self.rows = []
        self.closed = False
        self.flushed = False

    def append(self, row):
        self.rows.append(row)

    def close(self):
        self.closed = True

    def flush(self):
        self.flushed = True


def test_ana_base_validates_configuration():
    with pytest.raises(TypeError, match="obj_type"):
        DummyAna(obj_type=1)

    with pytest.raises(ValueError, match="run_mode"):
        DummyAna(run_mode="bad")

    with pytest.raises(ValueError, match="Object type"):
        DummyAna(obj_type="bad")

    with pytest.raises(ValueError, match="truth_point_mode"):
        DummyAna(truth_point_mode="bad")

    with pytest.raises(ValueError, match="incompatible"):
        DummyAna(truth_point_mode="points_adapt", truth_dep_mode="depositions_g4")

    with pytest.raises(ValueError, match="truth_dep_mode"):
        DummyAna(truth_dep_mode="bad")

    with pytest.raises(ValueError, match="non-empty"):
        DummyAna().initialize_writer("")


def test_ana_base_filters_entry_and_manages_writers(monkeypatch):
    monkeypatch.setattr(base_mod, "CSVWriter", DummyWriter)
    ana = DummyAna(log_dir="logs", prefix="prefix", append=True, overwrite=True)
    ana.update_keys({"value": True})
    ana.initialize_writer("out")

    result = ana(
        {
            "index": [5],
            "file_index": [2],
            "run_info": [DummyRunInfo()],
            "value": [9],
        },
        entry=0,
    )
    ana.flush_writers()
    ana.close_writers()

    writer = ana.writers["out"]
    assert result == {"updated": 10}
    assert writer.file_name == "logs/prefix_dummy_out.csv"
    assert writer.rows == [{"index": 5, "file_index": 2, "run": 1, "value": 9}]
    assert writer.flushed
    assert writer.closed


def test_ana_base_optional_base_fields_and_truth_accessors():
    ana = DummyAna(truth_point_mode="points", truth_index_mode="custom_index")

    class TruthObject:
        is_truth = True
        points = "truth_points"
        custom_index = "truth_index"

    base = ana.get_base_dict(
        {
            "index": 1,
            "file_index": 2,
            "file_entry_index": 3,
            "run_info": DummyRunInfo(),
        }
    )

    assert base == {"index": 1, "file_index": 2, "file_entry_index": 3, "run": 1}
    assert ana.get_points(TruthObject()) == "truth_points"
    assert ana.get_index(TruthObject()) == "truth_index"


def test_ana_base_warns_without_run_info_and_reads_reco_index():
    ana = DummyAna()

    class RecoObject:
        is_truth = False
        index = [1, 2, 3]

    with pytest.warns(UserWarning, match="run_info"):
        base = ana.get_base_dict({"index": 1, "file_index": 2})

    assert base == {"index": 1, "file_index": 2}
    assert ana.get_index(RecoObject()) == [1, 2, 3]


def test_ana_base_reports_missing_required_input():
    ana = DummyAna()
    ana.update_keys({"value": True})

    with pytest.raises(KeyError, match="missing an essential"):
        ana({"index": 0, "file_index": 0})
