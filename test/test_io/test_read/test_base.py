"""Tests for the generic reader base class."""

from pathlib import Path

import numpy as np
import pytest

from spine.io.read.base import ReaderBase


class DummyReader(ReaderBase):
    """Minimal ReaderBase implementation for unit tests."""

    name = "dummy"

    def __init__(self, num_entries=5, run_info=None):
        self.num_entries = num_entries
        self.entry_index = np.arange(num_entries, dtype=np.int64)
        self.file_index = np.zeros(num_entries, dtype=np.int64)
        self.file_offsets = np.asarray([0], dtype=np.int64)
        self.file_paths = ["file0"]
        self.run_info = run_info
        self.run_map = None

    def get(self, idx):
        return {"index": int(self.entry_index[idx])}


def test_reader_base_getters():
    """ReaderBase convenience accessors should use the configured indexes."""
    reader = DummyReader(
        num_entries=3,
        run_info=[(1, 0, 0), (1, 0, 1), (1, 0, 2)],
    )
    reader.process_run_info()

    assert len(reader) == 3
    assert reader[1] == {"index": 1}
    assert reader.get_run_event(1, 0, 2) == {"index": 2}
    assert reader.get_file_path(0) == "file0"
    assert reader.get_file_index(2) == 0
    assert reader.get_file_entry_index(2) == 2


def test_reader_base_process_file_paths(tmp_path):
    """ReaderBase should parse direct paths, text file lists, and remote URIs."""
    first = tmp_path / "a.h5"
    second = tmp_path / "b.h5"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"{second}\n{first}\n", encoding="utf-8")

    reader = DummyReader()
    reader.process_file_paths(file_keys=str(tmp_path / "*.h5"), limit_num_files=1)
    assert len(reader.file_paths) == 1

    reader.process_file_paths(file_list=str(file_list))
    assert reader.file_paths == sorted([str(first), str(second)])

    reader.process_file_paths(file_keys="root://server/file.root")
    assert reader.file_paths == ["root://server/file.root"]
    assert ReaderBase.is_remote_path("xroot://server/file.root")
    assert not ReaderBase.is_remote_path(str(first))


def test_reader_base_process_file_paths_errors(tmp_path):
    """ReaderBase should reject invalid file path inputs."""
    reader = DummyReader()
    with pytest.raises(AssertionError, match="either `file_keys` or `file_list`"):
        reader.process_file_paths()
    with pytest.raises(AssertionError, match="larger than 0"):
        reader.process_file_paths(file_keys=[], limit_num_files=0)
    with pytest.raises(AssertionError, match="valid path to a text file"):
        reader.process_file_paths(file_list=str(tmp_path / "missing.txt"))
    with pytest.raises(AssertionError, match="yielded no compatible path"):
        reader.process_file_paths(file_keys=str(tmp_path / "*.missing"))


def test_reader_base_process_run_info_and_entry_selection(tmp_path):
    """ReaderBase should build run maps and support entry selection helpers."""
    reader = DummyReader(
        num_entries=5,
        run_info=[(1, 0, i) for i in range(5)],
    )
    reader.process_run_info()
    assert reader.run_map[(1, 0, 3)] == 3

    reader.process_entry_list(n_entry=2, n_skip=1)
    assert np.array_equal(reader.entry_index, np.asarray([1, 2]))

    reader = DummyReader(
        num_entries=5,
        run_info=[(1, 0, i) for i in range(5)],
    )
    reader.process_run_info()
    reader.process_entry_list(entry_list=[0, 3])
    assert np.array_equal(reader.entry_index, np.asarray([0, 3]))

    reader = DummyReader(
        num_entries=5,
        run_info=[(1, 0, i) for i in range(5)],
    )
    reader.process_run_info()
    reader.process_entry_list(skip_entry_list=[1, 4])
    assert np.array_equal(reader.entry_index, np.asarray([0, 2, 3]))

    run_event_file = tmp_path / "run_events.txt"
    run_event_file.write_text("1 0 1\n1,0,3\n", encoding="utf-8")
    reader = DummyReader(
        num_entries=5,
        run_info=[(1, 0, i) for i in range(5)],
    )
    reader.process_run_info()
    reader.process_entry_list(run_event_list=str(run_event_file))
    assert np.array_equal(reader.entry_index, np.asarray([1, 3]))

    reader = DummyReader(
        num_entries=5,
        run_info=[(1, 0, i) for i in range(5)],
    )
    reader.process_run_info()
    reader.process_entry_list(skip_run_event_list=[(1, 0, 1), (1, 0, 3)])
    assert np.array_equal(reader.entry_index, np.asarray([0, 2, 4]))


def test_reader_base_process_entry_list_errors():
    """ReaderBase should reject inconsistent entry selection inputs."""
    reader = DummyReader(num_entries=3, run_info=[(1, 0, i) for i in range(3)])
    reader.process_run_info()

    with pytest.raises(AssertionError, match="Cannot specify `n_entry`"):
        reader.process_entry_list(n_entry=1, entry_list=[0])
    with pytest.raises(AssertionError, match="Cannot specify both `entry_list`"):
        reader.process_entry_list(entry_list=[0], skip_entry_list=[1])
    with pytest.raises(AssertionError, match="Cannot specify both `run_event_list`"):
        reader.process_entry_list(
            run_event_list=[(1, 0, 0)], skip_run_event_list=[(1, 0, 1)]
        )
    with pytest.raises(AssertionError, match="Incompatibility between `n_entry`"):
        reader.process_entry_list(n_entry=5)
    with pytest.raises(IndexError, match="No entries selected"):
        reader.process_entry_list(skip_entry_list=[0, 1, 2])


def test_reader_base_parse_helpers(tmp_path):
    """ReaderBase should parse entry and run-event lists from lists and files."""
    entries_file = tmp_path / "entries.txt"
    entries_file.write_text("1 2\n3,4\n", encoding="utf-8")
    events_file = tmp_path / "events.txt"
    events_file.write_text("1 0 1\n2,0,3\n", encoding="utf-8")

    assert np.array_equal(
        ReaderBase.parse_entry_list(None), np.empty(0, dtype=np.int64)
    )
    assert np.array_equal(ReaderBase.parse_entry_list([1, 2]), np.asarray([1, 2]))
    assert np.array_equal(
        ReaderBase.parse_entry_list(str(entries_file)), np.asarray([1, 2, 3, 4])
    )
    assert ReaderBase.parse_run_event_list(None) == []
    assert ReaderBase.parse_run_event_list([[1, 0, 1]]) == [(1, 0, 1)]
    assert ReaderBase.parse_run_event_list(str(events_file)) == [(1, 0, 1), (2, 0, 3)]

    with pytest.raises(AssertionError, match="does not exist"):
        ReaderBase.parse_entry_list(str(tmp_path / "missing.txt"))
    with pytest.raises(ValueError, match="List format not recognized"):
        ReaderBase.parse_entry_list(1.5)
    with pytest.raises(AssertionError, match="three integers"):
        ReaderBase.parse_run_event_list([[1, 2]])
    with pytest.raises(AssertionError, match="does not exist"):
        ReaderBase.parse_run_event_list(str(tmp_path / "missing_events.txt"))
    bad_events_file = tmp_path / "bad_events.txt"
    bad_events_file.write_text("1 0\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="three integers"):
        ReaderBase.parse_run_event_list(str(bad_events_file))
    with pytest.raises(ValueError, match="List format not recognized"):
        ReaderBase.parse_run_event_list(1.5)


def test_reader_base_process_run_info_errors():
    """ReaderBase should reject invalid run information."""
    reader = DummyReader(num_entries=2, run_info=[(1, 0, 0)])
    with pytest.raises(AssertionError):
        reader.process_run_info()

    reader = DummyReader(num_entries=2, run_info=[(1, 0, 0), (1, 0, 0)])
    with pytest.raises(AssertionError, match="not unique"):
        reader.process_run_info()


def test_reader_base_get_run_event_index_errors():
    """ReaderBase should require a run map and known triplets."""
    reader = DummyReader(num_entries=1)
    with pytest.raises(AssertionError, match="Must build a run map"):
        reader.get_run_event_index(1, 0, 0)

    reader = DummyReader(num_entries=1, run_info=[(1, 0, 0)])
    reader.process_run_info()
    with pytest.raises(AssertionError, match="Could not find"):
        reader.get_run_event_index(1, 0, 1)
