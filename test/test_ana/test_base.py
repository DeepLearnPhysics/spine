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


class ColumnarDummyAna(DummyAna):
    def process_columnar(self, data):
        return {"updated": [value + 1 for value in data["value"]]}


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
    monkeypatch.setattr(base_mod, "CSVLogger", DummyWriter)
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
    ana = DummyAna(
        truth_point_mode="points",
        truth_index_mode="custom_index",
        truth_dep_mode="depositions",
    )

    class TruthObject:
        is_truth = True
        points = "truth_points"
        depositions = "truth_depositions"
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
    assert ana.get_depositions(TruthObject()) == "truth_depositions"
    assert ana.get_index(TruthObject()) == "truth_index"


def test_ana_base_warns_without_run_info_and_reads_reco_index():
    ana = DummyAna()

    class RecoObject:
        is_truth = False
        index = [1, 2, 3]
        depositions = "reco_depositions"

    with pytest.warns(UserWarning, match="run_info"):
        base = ana.get_base_dict({"index": 1, "file_index": 2})

    assert base == {"index": 1, "file_index": 2}
    assert ana.get_index(RecoObject()) == [1, 2, 3]
    assert ana.get_depositions(RecoObject()) == "reco_depositions"


def test_ana_base_reports_missing_required_input():
    ana = DummyAna()
    ana.update_keys({"value": True})

    with pytest.raises(KeyError, match="missing an essential"):
        ana({"index": 0, "file_index": 0})


def test_ana_base_dispatches_optional_columnar_hook():
    ana = ColumnarDummyAna()
    ana.update_keys({"value": True, "optional": False})
    data = {
        "index": [0, 1],
        "file_index": [2, 2],
        "value": [4, 8],
        "unrequested": [10, 20],
    }

    assert ana.supports_columnar
    assert ana.run_columnar(data) == {"updated": [5, 9]}


def test_ana_base_rejects_unsupported_columnar_execution():
    ana = DummyAna()

    assert not ana.supports_columnar
    with pytest.raises(NotImplementedError, match="does not implement"):
        ana.run_columnar({"index": [0], "file_index": [0]})
    with pytest.raises(NotImplementedError):
        ana.process_columnar({})


def test_ana_base_reports_missing_required_columnar_inputs():
    """Columnar filtering should validate administrative and product inputs."""
    ana = ColumnarDummyAna()
    ana.update_keys({"value": True})

    with pytest.raises(KeyError, match="`index`"):
        ana.run_columnar({"file_index": [0], "value": [1]})

    with pytest.raises(KeyError, match="`value`"):
        ana.run_columnar({"index": [0], "file_index": [0]})
