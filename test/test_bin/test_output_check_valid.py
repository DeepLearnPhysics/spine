"""Tests for the output validation helper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


def load_output_check_valid_module():
    """Import ``bin/output_check_valid.py`` as a test module."""
    script_path = Path(__file__).resolve().parents[2] / "bin" / "output_check_valid.py"
    spec = importlib.util.spec_from_file_location("output_check_valid", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_events_file(
    path: Path, num_entries: int, complete: bool | None = None
) -> None:
    """Create a minimal HDF5 file with an ``events`` dataset."""
    dtype = np.dtype([("dummy", np.int64)])
    with h5py.File(path, "w") as out_file:
        info = out_file.create_group("info")
        if complete is not None:
            info.attrs["complete"] = complete
        out_file.create_dataset("events", data=np.zeros(num_entries, dtype=dtype))


def add_source_group(path: Path, source_path: Path) -> None:
    """Attach modern top-level source provenance to an existing HDF5 file."""
    stat_result = source_path.stat()
    with h5py.File(path, "a") as out_file:
        source = out_file.create_group("source")
        source.attrs["file_name"] = source_path.name
        source.attrs["file_size"] = int(stat_result.st_size)
        source.attrs["file_mtime_ns"] = int(stat_result.st_mtime_ns)


def test_get_num_entries_hdf5(tmp_path):
    """The helper should count HDF5 entries from the events dataset."""
    module = load_output_check_valid_module()
    path = tmp_path / "input.h5"
    make_events_file(path, 3)

    assert module.get_num_entries(str(path)) == 3


def test_has_modern_hdf5_markers(tmp_path):
    """Modern metadata markers should be detected from either source or complete."""
    module = load_output_check_valid_module()
    legacy = tmp_path / "legacy.h5"
    modern_complete = tmp_path / "modern_complete.h5"
    modern_source = tmp_path / "modern_source.h5"
    source_file = tmp_path / "source.h5"
    source_file.write_bytes(b"source")

    make_events_file(legacy, 1)
    make_events_file(modern_complete, 1, complete=True)
    make_events_file(modern_source, 1)
    add_source_group(modern_source, source_file)

    with h5py.File(legacy, "r") as out_file:
        assert not module.has_modern_hdf5_markers(out_file)
    with h5py.File(modern_complete, "r") as out_file:
        assert module.has_modern_hdf5_markers(out_file)
    with h5py.File(modern_source, "r") as out_file:
        assert module.has_modern_hdf5_markers(out_file)


def test_check_hdf5_source_provenance(tmp_path):
    """Source provenance should validate basename, size, and mtime."""
    module = load_output_check_valid_module()
    source_path = tmp_path / "input.h5"
    source_path.write_bytes(b"source")
    output_path = tmp_path / "output.h5"
    make_events_file(output_path, 1)
    add_source_group(output_path, source_path)

    with h5py.File(output_path, "r") as out_file:
        assert module.check_hdf5_source_provenance(str(source_path), out_file)

    with h5py.File(output_path, "a") as out_file:
        out_file["source"].attrs["file_size"] = int(source_path.stat().st_size) + 1

    with h5py.File(output_path, "r") as out_file:
        assert not module.check_hdf5_source_provenance(str(source_path), out_file)


def test_is_valid_modern_hdf5_output(tmp_path):
    """Modern HDF5 validation should distinguish valid, invalid, and legacy outputs."""
    module = load_output_check_valid_module()
    source_path = tmp_path / "input.h5"
    source_path.write_bytes(b"source")

    legacy = tmp_path / "legacy.h5"
    valid = tmp_path / "valid.h5"
    incomplete = tmp_path / "incomplete.h5"
    mismatched = tmp_path / "mismatched.h5"

    make_events_file(legacy, 1)
    make_events_file(valid, 1, complete=True)
    add_source_group(valid, source_path)
    make_events_file(incomplete, 1, complete=False)
    add_source_group(incomplete, source_path)
    make_events_file(mismatched, 1, complete=True)
    add_source_group(mismatched, source_path)
    with h5py.File(mismatched, "a") as out_file:
        out_file["source"].attrs["file_name"] = "other.h5"

    assert module.is_valid_modern_hdf5_output(str(source_path), str(legacy)) is None
    assert module.is_valid_modern_hdf5_output(str(source_path), str(valid)) is True
    assert (
        module.is_valid_modern_hdf5_output(str(source_path), str(incomplete)) is False
    )
    assert (
        module.is_valid_modern_hdf5_output(str(source_path), str(mismatched)) is False
    )


def test_main_accepts_modern_hdf5_output_even_if_counts_differ(tmp_path):
    """Modern HDF5 validation should prefer completeness metadata over counts."""
    module = load_output_check_valid_module()
    source_path = tmp_path / "input.h5"
    dest = tmp_path / "outputs"
    output_list = tmp_path / "bad.txt"
    dest.mkdir()

    make_events_file(source_path, 2)
    output_path = dest / "input_spine.h5"
    make_events_file(output_path, 1, complete=True)
    add_source_group(output_path, source_path)

    module.main(
        source=[str(source_path)],
        source_list=None,
        output=str(output_list),
        dest=str(dest),
        suffix="spine",
        event_list=None,
        tree_name=None,
        larcv_output=False,
    )

    assert output_list.read_text(encoding="utf-8") == ""


def test_main_falls_back_to_legacy_entry_count_check(tmp_path):
    """Legacy HDF5 outputs should still be checked by entry count."""
    module = load_output_check_valid_module()
    source_path = tmp_path / "input.h5"
    dest = tmp_path / "outputs"
    output_list = tmp_path / "bad.txt"
    dest.mkdir()

    make_events_file(source_path, 3)
    output_path = dest / "input_spine.h5"
    make_events_file(output_path, 1)

    module.main(
        source=[str(source_path)],
        source_list=None,
        output=str(output_list),
        dest=str(dest),
        suffix="spine",
        event_list=None,
        tree_name=None,
        larcv_output=False,
    )

    assert output_list.read_text(encoding="utf-8").strip() == str(source_path)


def test_main_reports_missing_output_file(tmp_path):
    """Missing outputs should be written to the retry list."""
    module = load_output_check_valid_module()
    source_path = tmp_path / "input.h5"
    dest = tmp_path / "outputs"
    output_list = tmp_path / "bad.txt"
    dest.mkdir()

    make_events_file(source_path, 1)

    module.main(
        source=[str(source_path)],
        source_list=None,
        output=str(output_list),
        dest=str(dest),
        suffix="spine",
        event_list=None,
        tree_name=None,
        larcv_output=False,
    )

    assert output_list.read_text(encoding="utf-8").strip() == str(source_path)
