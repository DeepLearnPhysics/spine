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
        self.calls: list[tuple[str, str]] = []
        self.watches: dict[str, SimpleNamespace] = {}

    def initialize(self, key: str) -> None:
        self.initialized.append(key)
        self.watches.setdefault(key, SimpleNamespace(running=False, paused=False))

    def start(self, key: str) -> None:
        self.calls.append(("start", key))
        self.watches.setdefault(key, SimpleNamespace(running=False, paused=False))
        self.watches[key].running = True

    def stop(self, key: str) -> None:
        self.calls.append(("stop", key))
        self.watches.setdefault(key, SimpleNamespace(running=False, paused=False))
        self.watches[key].running = False

    def reset(self) -> None:
        self.calls.append(("reset", None))
        for watch in self.watches.values():
            watch.running = False
            watch.paused = False

    def reset_if_active(self) -> None:
        for watch in self.watches.values():
            if watch.running or watch.paused:
                self.reset()
                break

    def values(self):
        return self.watches.values()


@pytest.fixture(autouse=True)
def fixture_fake_stopwatch_manager(monkeypatch):
    """Use a lightweight stopwatch manager in IOManager tests."""
    monkeypatch.setattr(manager_mod, "StopwatchManager", FakeWatch)


class FakeReader:
    """Reader-like object used by IOManager tests."""

    file_paths = ["/tmp/input_a.root", "/tmp/input_b.root"]
    cfg = {"post": {"existing": {}}}

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __len__(self) -> int:
        return 4

    def get(self, entry: int) -> dict[str, int]:
        self.calls.append(("get", entry))
        return {"index": entry}

    def get_run_event(self, run: int, subrun: int, event: int) -> dict[str, int]:
        self.calls.append(("get_run_event", (run, subrun, event)))
        return {"index": event}

    def process_entry_list(self, *args: object) -> None:
        self.calls.append(("process_entry_list", args))


class FakeLoader:
    """Loader-like object used by IOManager tests."""

    def __init__(self) -> None:
        self.dataset = SimpleNamespace(reader=FakeReader())
        self.batches = iter([{"index": [0, 1]}, {"index": [2, 3]}])
        self.sampler = SimpleNamespace(epochs=[])
        self.sampler.set_epoch = lambda epoch: self.sampler.epochs.append(epoch)

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        self.batches = iter([{"index": [0, 1]}, {"index": [2, 3]}])
        return self

    def __next__(self) -> dict[str, list[int]]:
        return next(self.batches)


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
        iterations=-1,
        split_output=False,
    )

    assert manager.loader is None
    assert not manager.has_loader
    assert manager.reader.file_paths == ["/tmp/input_a.root", "/tmp/input_b.root"]
    assert len(manager) == 4
    assert manager.post_list == ("existing",)
    assert manager.iterations == 4
    assert manager.epochs == 1.0
    assert manager.writer == "writer"
    assert manager.has_writer
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
        rank=1,
        dtype="float64",
        world_size=2,
        distributed=True,
        unwrap=True,
        epochs=1.5,
        split_output=True,
    )

    assert manager.loader is not None
    assert manager.has_loader
    assert manager.unwrapper == "unwrapper"
    assert manager.post_list == ()
    assert manager.iterations == 3
    assert manager.output_prefix == ["input_a", "input_b"]
    assert calls[0]["rank"] == 1
    assert calls[0]["dtype"] == "float64"
    assert calls[0]["distributed"] is True


def test_io_manager_allows_on_demand_iteration_config(monkeypatch):
    """IOManager should allow omitted iteration bounds for on-demand loading."""
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: FakeReader())

    manager = IOManager(reader={"name": "hdf5"})

    assert manager.iterations is None
    assert manager.epochs is None
    assert manager.iter_per_epoch == 4


def test_io_manager_validation(monkeypatch):
    """IOManager should reject invalid I/O combinations."""
    with pytest.raises(ValueError, match="either a loader or a reader"):
        IOManager()

    with pytest.raises(ValueError, match="either a loader or a reader"):
        IOManager(loader={}, reader={})

    with pytest.raises(ValueError, match="iterations"):
        IOManager(reader={}, iterations=1, epochs=1)

    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: FakeReader())
    with pytest.raises(ValueError, match=r"base\.split_output: true"):
        IOManager(reader={}, writer={"name": "stage_hdf5"}, split_output=False)

    monkeypatch.setattr(manager_mod, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError, match="loader"):
        IOManager(loader={}, epochs=1.0)

    monkeypatch.setattr(manager_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(manager_mod, "loader_factory", lambda **kwargs: FakeLoader())
    with pytest.raises(ValueError, match="write"):
        IOManager(loader={}, writer={}, unwrap=False, epochs=1.0)

    monkeypatch.setattr(
        manager_mod, "loader_factory", lambda **kwargs: FakeLoaderNoReader()
    )
    with pytest.raises(RuntimeError, match="reader"):
        IOManager(loader={}, epochs=1.0)

    manager = object.__new__(IOManager)
    manager.reader = None
    with pytest.raises(RuntimeError, match="length"):
        len(manager)

    manager.watch = FakeWatch()
    with pytest.raises(RuntimeError, match="Reader configuration"):
        manager._initialize_reader(None)


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

    manager.log_prefix = "input"
    monkeypatch.setattr(manager, "_name_max", lambda path=".": 20)
    assert manager.format_log_name("spine_log.csv", ".") == "input_spine_log.csv"


def test_io_manager_load_reader_paths(monkeypatch):
    """IOManager.load should dispatch reader entry and run-event requests."""
    reader = FakeReader()
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: reader)

    manager = IOManager(reader={"name": "hdf5"}, iterations=-1)

    assert manager.load(entry=1) == {"index": 1}
    assert manager.load(run=1, subrun=2, event=3) == {"index": 3}
    assert reader.calls == [("get", 1), ("get_run_event", (1, 2, 3))]
    assert ("start", "read") in manager.watch.calls
    assert ("stop", "read") in manager.watch.calls

    with pytest.raises(ValueError, match="entry number"):
        manager.load()

    manager.reader = None
    with pytest.raises(RuntimeError, match="reader"):
        manager.load(entry=1)


def test_io_manager_load_loader_paths(monkeypatch):
    """IOManager.load should own sequential loader access."""
    monkeypatch.setattr(manager_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(manager_mod, "loader_factory", lambda **kwargs: FakeLoader())

    manager = IOManager(loader={"dataset": {}}, epochs=1.0)

    assert manager.load() == {"index": [0, 1]}
    assert manager.load() == {"index": [2, 3]}
    assert ("start", "load") in manager.watch.calls
    assert ("stop", "load") in manager.watch.calls

    with pytest.raises(ValueError, match="specific entry"):
        manager.load(entry=0)


def test_io_manager_resets_stale_watch_before_timed_operation(monkeypatch):
    """IOManager should clear its own active watch before a new timed call."""
    reader = FakeReader()
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: reader)

    manager = IOManager(reader={"name": "hdf5"}, iterations=-1)
    manager.watch.calls.clear()
    manager.watch.start("read")

    assert manager.load(entry=1) == {"index": 1}
    assert manager.watch.calls[:3] == [
        ("start", "read"),
        ("reset", None),
        ("start", "read"),
    ]
    assert manager.watch.calls[-1] == ("stop", "read")


def test_io_manager_iteration_unwrap_write_and_close(monkeypatch):
    """IOManager should own loader iteration, unwrapping and writer lifecycle."""
    writer_calls: list[object] = []

    class FakeWriter:
        def __call__(self, data, cfg):
            writer_calls.append(("write", data, cfg))

        def finalize(self):
            writer_calls.append("finalize")

        def close(self):
            writer_calls.append("close")

    monkeypatch.setattr(manager_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(manager_mod, "loader_factory", lambda **kwargs: FakeLoader())
    monkeypatch.setattr(
        manager_mod, "Unwrapper", lambda: lambda data: {"index": [data["index"]]}
    )
    monkeypatch.setattr(
        manager_mod, "writer_factory", lambda *args, **kwargs: FakeWriter()
    )

    manager = IOManager(
        loader={"dataset": {}},
        writer={"name": "hdf5"},
        distributed=True,
        unwrap=True,
        epochs=1.0,
    )

    assert manager.loader_iter is None
    manager.prepare_iteration(0)
    assert manager.loader_iter is not None
    assert manager.loader.sampler.epochs == [0]

    loader_iter = manager.loader_iter
    manager.prepare_iteration(1)
    assert manager.loader_iter is loader_iter
    manager.prepare_iteration(2)
    assert manager.loader.sampler.epochs == [0, 1]

    assert manager.unwrap({"index": [0]}) == {"index": [[0]]}
    assert ("start", "unwrap") in manager.watch.calls
    assert ("stop", "unwrap") in manager.watch.calls

    manager.write({"index": 0}, {"cfg": True})
    manager.close()
    assert writer_calls == [
        ("write", {"index": 0}, {"cfg": True}),
        "finalize",
        "close",
    ]

    manager.writer = None
    assert not manager.has_writer
    manager.write({"index": 1}, {})
    manager.close()

    manager.unwrapper = None
    data = {"index": 1}
    assert manager.unwrap(data) is data

    manager.loader = None
    manager.loader_iter = None
    manager.prepare_iteration(0)
    assert manager.loader_iter is None


def test_io_manager_apply_filter(monkeypatch):
    """IOManager.apply_filter should delegate to the reader and reset loaders."""
    reader = FakeReader()
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: reader)

    manager = IOManager(reader={"name": "hdf5"}, iterations=-1)
    manager.loader_iter = object()
    manager.apply_filter(1, 2, [3], [4], [(1, 2, 3)], [(4, 5, 6)])

    assert reader.calls == [
        ("process_entry_list", (1, 2, [3], [4], [(1, 2, 3)], [(4, 5, 6)]))
    ]
    assert manager.loader_iter is None

    manager.reader = None
    with pytest.raises(RuntimeError, match="reader"):
        manager.apply_filter()
