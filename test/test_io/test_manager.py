"""Tests for the SPINE I/O manager."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import spine.io.manager as manager_mod
from spine.io import IOManager


class FakeWatch:
    """Minimal timer registry used by IOManager tests."""

    def __init__(self) -> None:
        self.initialized: list[str] = []

    def initialize(self, key: str) -> None:
        self.initialized.append(key)


class FakeReader:
    """Reader-like object used by IOManager tests."""

    file_paths = ["/tmp/input_a.root", "/tmp/input_b.root"]
    cfg = {"post": {"existing": {}}}

    def __len__(self) -> int:
        return 4


class FakeLoader:
    """Loader-like object used by IOManager tests."""

    def __init__(self) -> None:
        self.dataset = SimpleNamespace(reader=FakeReader())

    def __len__(self) -> int:
        return 2


class FakeLoaderNoReader:
    """Loader-like object with an invalid dataset reader."""

    def __init__(self) -> None:
        self.dataset = SimpleNamespace(reader=None)

    def __len__(self) -> int:
        return 1


def test_io_manager_initializes_reader_writer_and_iterations(monkeypatch):
    """Reader setup should derive prefixes, writer and iteration count."""
    writer_calls: list[tuple[object, str | list[str], bool]] = []
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: FakeReader())
    monkeypatch.setattr(
        manager_mod,
        "writer_factory",
        lambda cfg, prefix, split: writer_calls.append((cfg, prefix, split))
        or "writer",
    )

    manager = IOManager(
        reader={"name": "hdf5"},
        writer={"name": "hdf5"},
        watch=FakeWatch(),
        iterations=-1,
        split_output=False,
    )

    assert manager.loader is None
    assert manager.reader.file_paths == ["/tmp/input_a.root", "/tmp/input_b.root"]
    assert manager.post_list == ("existing",)
    assert manager.iterations == 4
    assert manager.epochs == 1.0
    assert manager.writer == "writer"
    assert writer_calls == [({"name": "hdf5"}, "input_a--input_b", False)]


def test_io_manager_initializes_loader_and_unwrapper(monkeypatch):
    """Loader setup should pass through runtime context and optional unwrap."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(manager_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        manager_mod,
        "loader_factory",
        lambda **kwargs: calls.append(kwargs) or FakeLoader(),
    )
    monkeypatch.setattr(manager_mod, "Unwrapper", lambda: "unwrapper")

    manager = IOManager(
        loader={"dataset": {}},
        watch=FakeWatch(),
        geo={"detector": "icarus"},
        rank=1,
        dtype="float64",
        world_size=2,
        distributed=True,
        unwrap=True,
        epochs=1.5,
        split_output=True,
    )

    assert manager.loader is not None
    assert manager.unwrapper == "unwrapper"
    assert manager.post_list == ()
    assert manager.iterations == 3
    assert manager.output_prefix == ["input_a", "input_b"]
    assert calls[0]["geo"] == {"detector": "icarus"}
    assert calls[0]["rank"] == 1
    assert calls[0]["dtype"] == "float64"
    assert calls[0]["distributed"] is True


def test_io_manager_validation(monkeypatch):
    """IOManager should reject invalid I/O combinations."""
    with pytest.raises(ValueError, match="either a loader or a reader"):
        IOManager(watch=FakeWatch())

    with pytest.raises(ValueError, match="either a loader or a reader"):
        IOManager(loader={}, reader={}, watch=FakeWatch())

    with pytest.raises(ValueError, match="iterations"):
        IOManager(reader={}, watch=FakeWatch(), iterations=1, epochs=1)

    monkeypatch.setattr(manager_mod, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError, match="loader"):
        IOManager(loader={}, watch=FakeWatch())

    monkeypatch.setattr(manager_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(manager_mod, "loader_factory", lambda **kwargs: FakeLoader())
    with pytest.raises(ValueError, match="write"):
        IOManager(loader={}, writer={}, watch=FakeWatch(), unwrap=False)

    monkeypatch.setattr(
        manager_mod, "loader_factory", lambda **kwargs: FakeLoaderNoReader()
    )
    with pytest.raises(RuntimeError, match="reader"):
        IOManager(loader={}, watch=FakeWatch())


def test_io_manager_prefix_variants(monkeypatch):
    """Prefix helper should cover single, duplicate, skipped and long names."""
    manager = object.__new__(IOManager)
    monkeypatch.setattr(
        manager_mod.os,
        "pathconf",
        lambda *args: (_ for _ in ()).throw(OSError()),
    )
    assert manager._name_max() == 255
    assert manager._truncate_prefix("abcdef", 3) == "---"

    assert manager.get_prefixes(["/tmp/file.root"], False) == ("file", "file")
    assert manager.get_prefixes(["/tmp/file.root"], True) == ("file", ["file"])
    assert manager.get_prefixes(["same.root", "same.root"], False) == (
        "same",
        "same",
    )
    assert manager.get_prefixes(["a_001.root", "a_002.root", "a_003.root"], True) == (
        "a_001--3files--a_003",
        ["a_001", "a_002", "a_003"],
    )
    assert manager.get_prefixes(
        ["prefix_a_tail.root", "prefix_b_tail.root"], False
    ) == (
        "prefix_a_tail--prefix_b_tail",
        "prefix_a_tail--prefix_b_tail",
    )
    with pytest.raises(ValueError, match="at least one"):
        manager.get_prefixes([], False)

    monkeypatch.setattr(manager, "_name_max", lambda: 80)
    long_names = [f"very_long_prefix_{'a' * 200}_{idx}.root" for idx in range(2)]
    log_prefix, output_prefix = manager.get_prefixes(long_names, False)
    assert len(log_prefix) == 80
    assert "---" in log_prefix
    assert output_prefix == log_prefix

    log_prefix, output_prefix = manager.get_prefixes(
        long_names, True, output_suffix="_custom.h5"
    )
    assert len(log_prefix) == 80
    assert all(len(prefix) == 70 for prefix in output_prefix)
