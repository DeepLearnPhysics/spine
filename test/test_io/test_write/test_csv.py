"""Tests for the CSV writer."""

from pathlib import Path

import pytest

from spine.io.write.csv import CSVWriter


def test_csv_writer_context_manager_and_flush(tmp_path):
    """CSVWriter should open, flush, and close cleanly through the context manager."""
    path = tmp_path / "output.csv"

    with CSVWriter(path, buffer_size=1) as writer:
        writer.append({"a": 1, "b": 2})
        writer.flush()
        assert writer.file_handle is not None

    assert writer.file_handle is None
    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2"]


def test_csv_writer_rejects_existing_file_without_overwrite(tmp_path):
    """CSVWriter should protect existing files by default."""
    path = tmp_path / "output.csv"
    path.write_text("a,b\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        CSVWriter(path)


def test_csv_writer_append_mode_requires_existing_file(tmp_path):
    """CSVWriter append mode should require a file with a header."""
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="must exist"):
        CSVWriter(path, append=True)


def test_csv_writer_append_mode_reads_existing_header(tmp_path):
    """CSVWriter append mode should preserve the original header order."""
    path = tmp_path / "output.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    writer = CSVWriter(path, append=True)
    writer.append({"a": 3, "b": 4})
    writer.close()

    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2", "3,4"]


def test_csv_writer_rejects_excess_keys(tmp_path):
    """CSVWriter should reject rows with unexpected keys."""
    path = tmp_path / "output.csv"
    writer = CSVWriter(path)
    writer.append({"a": 1})

    with pytest.raises(AssertionError, match="New keys"):
        writer.append({"a": 2, "b": 3})

    writer.close()


def test_csv_writer_handles_missing_keys_when_allowed(tmp_path):
    """CSVWriter should backfill missing values when configured to allow them."""
    path = tmp_path / "output.csv"
    writer = CSVWriter(path, accept_missing=True)
    writer.append({"a": 1, "b": 2})
    writer.append({"a": 3})
    writer.close()

    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2", "3,-1"]


def test_csv_writer_rejects_missing_keys_by_default(tmp_path):
    """CSVWriter should reject missing keys unless explicitly allowed."""
    path = tmp_path / "output.csv"
    writer = CSVWriter(path)
    writer.append({"a": 1, "b": 2})

    with pytest.raises(AssertionError, match="Missing keys"):
        writer.append({"a": 3})

    writer.close()


def test_csv_writer_array_diff():
    """CSVWriter.array_diff should return elements missing from the second array."""
    diff = CSVWriter.array_diff(["a", "b", "c"], ["b"])
    assert set(diff) == {"a", "c"}


def test_csv_writer_directory_relocates_output(tmp_path):
    """CSVWriter should support relocating output under an explicit directory."""
    out_dir = tmp_path / "logs"
    writer = CSVWriter("output.csv", directory=str(out_dir), overwrite=True)
    assert writer.file_name == str(out_dir / "output.csv")
