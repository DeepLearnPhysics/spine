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


class FakeColumnarReader(FakeReader):
    """Reader-like object which exposes two projected chunks."""

    columnar = True
    num_chunks = 2

    def configure_columnar(self, requests):
        self.calls.append(("configure_columnar", requests))

    def get_columnar(self, entry):
        self.calls.append(("get_columnar", entry))
        return {"index": [2 * entry, 2 * entry + 1]}


class FakeStageReader(FakeReader):
    """Reader-like staged cache used to exercise same-file writer routing."""

    name = "stage_hdf5"
    file_paths = ["/tmp/cache_a.h5", "/tmp/cache_b.h5"]


class FakeLoader:
    """Loader-like object used by IOManager tests."""

    def __init__(self) -> None:
        self.dataset = SimpleNamespace(reader=FakeReader())
        self.batches = iter([{"index": [0, 1]}, {"index": [2, 3]}])
        self.batch_size = 2
        self.num_workers = 0
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


class FakeMixedLoader(FakeLoader):
    """Loader-like object with primary LArCV and staged-cache readers."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset = SimpleNamespace(
            reader=FakeReader(),
            cache=SimpleNamespace(reader=FakeStageReader()),
        )


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


def test_io_manager_merges_cumulative_post_provenance(monkeypatch):
    """Reader provenance should supersede cfg and extend without duplicates."""
    reader = FakeReader()
    reader.post_processors = ("first", "shared")
    recorded: list[tuple[str, ...]] = []
    writer = SimpleNamespace(set_post_processors=recorded.append)
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: reader)
    monkeypatch.setattr(manager_mod, "writer_factory", lambda *args, **kwargs: writer)

    manager = IOManager(
        reader={"name": "hdf5"},
        writer={"name": "hdf5"},
    )
    manager.set_post_processors(("shared", "third"))

    assert manager.post_list == ("first", "shared", "third")
    assert recorded == [("first", "shared", "third")]


@pytest.mark.parametrize(
    ("post_cfg", "expected"),
    [
        ({"custom_alias": {"name": "canonical"}}, ("canonical",)),
        ({"plain": None}, ("plain",)),
        (["first", "second"], ("first", "second")),
    ],
)
def test_io_manager_canonicalizes_legacy_post_provenance(
    monkeypatch, post_cfg, expected
):
    """Legacy cfg fallback should seed provenance with canonical names."""
    reader = FakeReader()
    reader.cfg = {"post": post_cfg}
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: reader)

    manager = IOManager(reader={"name": "hdf5"})

    assert manager.post_list == expected


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

    manager.reader = FakeReader()
    manager.columnar = False
    with pytest.raises(RuntimeError, match="not configured"):
        manager.configure_columnar({"value": (("id",), True)})


def test_io_manager_uses_sidecars_for_same_file_staged_writes(monkeypatch):
    """Staged input/output jobs should receive automatic sidecar routing."""
    writer_calls = []
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: FakeStageReader())

    def build_writer(cfg, **kwargs):
        writer_calls.append((cfg, kwargs))
        return object()

    monkeypatch.setattr(manager_mod, "writer_factory", build_writer)
    IOManager(
        reader={"name": "stage_hdf5"},
        writer={"name": "stage_hdf5", "stage": "downstream"},
        split_output=True,
    )

    writer_cfg, kwargs = writer_calls[0]
    assert writer_cfg["sidecar"] is True
    assert writer_cfg["target_file_paths"] == FakeStageReader.file_paths
    assert kwargs["split"] is True

    IOManager(
        reader={"name": "stage_hdf5"},
        writer={
            "name": "stage_hdf5",
            "stage": "downstream",
            "file_name": "/tmp/separate.h5",
        },
        split_output=True,
    )
    writer_cfg, _ = writer_calls[1]
    assert "sidecar" not in writer_cfg
    assert "target_file_paths" not in writer_cfg

    IOManager(
        reader={"name": "stage_hdf5"},
        writer={
            "name": "stage_hdf5",
            "stage": "downstream",
            "sidecar": False,
        },
        split_output=True,
    )
    writer_cfg, _ = writer_calls[2]
    assert writer_cfg["sidecar"] is False
    assert "target_file_paths" not in writer_cfg


def test_io_manager_uses_mixed_dataset_cache_reader_for_sidecars(monkeypatch):
    """Mixed loaders should extend their staged cache, not the LArCV source."""
    writer_calls = []
    monkeypatch.setattr(manager_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        manager_mod, "loader_factory", lambda **kwargs: FakeMixedLoader()
    )
    monkeypatch.setattr(
        manager_mod,
        "writer_factory",
        lambda cfg, **kwargs: writer_calls.append((cfg, kwargs)) or object(),
    )

    manager = IOManager(
        loader={"dataset": {"name": "mixed"}},
        writer={"name": "stage_hdf5", "stage": "fragmentation"},
        unwrap=True,
        split_output=True,
    )

    writer_cfg, kwargs = writer_calls[0]
    assert manager.reader.file_paths == FakeReader.file_paths
    assert writer_cfg["sidecar"] is True
    assert writer_cfg["target_file_paths"] == FakeStageReader.file_paths
    assert kwargs["split"] is True


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


def test_io_manager_loads_and_configures_columnar_chunks(monkeypatch):
    reader = FakeColumnarReader()
    monkeypatch.setattr(manager_mod, "reader_factory", lambda cfg: reader)
    manager = IOManager(reader={"name": "hdf5"}, iterations=-1)
    requests = {"particles": (("id", "pid"), True)}

    assert manager.columnar
    assert manager.iter_per_epoch == 2
    assert len(manager) == 2
    manager.configure_columnar(requests)
    assert manager.load(entry=1) == {"index": [2, 3]}
    assert reader.calls == [
        ("configure_columnar", requests),
        ("get_columnar", 1),
    ]

    with pytest.raises(ValueError, match="chunk index"):
        manager.load(run=1, subrun=2, event=3)


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

    manager.loader.sampler.epochs.clear()
    manager.loader_iter = None
    manager.set_resume_progress(iteration=5, epoch=2.5)
    manager.prepare_iteration(5)
    manager.prepare_iteration(6)
    assert manager.loader.sampler.epochs == [2, 3]

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

    writer_calls.clear()
    manager.writer = FakeWriter()
    manager.close(finalize=False)
    assert writer_calls == ["close"]

    class FailingWriter(FakeWriter):
        def finalize(self):
            writer_calls.append("finalize")
            raise RuntimeError("finalize failed")

    writer_calls.clear()
    manager.writer = FailingWriter()
    with pytest.raises(RuntimeError, match="finalize failed"):
        manager.close()
    assert writer_calls == ["finalize", "close"]

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
    manager.apply_filter(entry_fraction_range=(0.25, 0.75))

    assert reader.calls == [
        (
            "process_entry_list",
            (1, 2, [3], [4], [(1, 2, 3)], [(4, 5, 6)], None),
        ),
        (
            "process_entry_list",
            (None, None, None, None, None, None, (0.25, 0.75)),
        ),
    ]
    assert manager.loader_iter is None

    manager.reader = None
    with pytest.raises(RuntimeError, match="reader"):
        manager.apply_filter()


def test_io_manager_checkpoints_and_restores_sampler_cursor():
    """Loader state should preserve the next batch and sampler epoch order."""

    class Sampler:
        def __init__(self):
            self.loaded = None

        @staticmethod
        def state_dict():
            return {"indices": [4, 5, 0, 1, 2, 3]}

        def load_state_dict(self, state, offset=0):
            self.loaded = (state, offset)

    manager = object.__new__(IOManager)
    manager.loader = SimpleNamespace(
        sampler=Sampler(),
        batch_size=2,
        num_workers=0,
    )
    manager.loader_iter = object()
    manager.iter_per_epoch = 3
    manager._resume_skip_batches = 0

    state = manager.checkpoint_state(next_iteration=5)
    manager.restore_checkpoint_state(state)

    assert state["batch_offset"] == 2
    assert manager.loader.sampler.loaded == (state["sampler"], 4)
    assert manager.loader_iter is None

    manager.loader = None
    assert manager.checkpoint_state(6) is None
    with pytest.raises(ValueError, match="without a loader"):
        manager.restore_checkpoint_state(state)


def test_io_manager_checkpoint_supports_parameterless_sampler_state():
    """Third-party sampler state methods need not accept SPINE's cursor."""

    class Sampler:
        @staticmethod
        def state_dict():
            return {"third_party": True}

    manager = object.__new__(IOManager)
    manager.loader = SimpleNamespace(
        sampler=Sampler(),
        batch_size=2,
        num_workers=0,
    )
    manager.iter_per_epoch = 3

    assert manager.checkpoint_state(1)["sampler"] == {"third_party": True}


def test_io_manager_replays_generic_sampler_cursor():
    """A generic loader should consume restored batches before yielding."""

    class Loader:
        def __iter__(self):
            return iter([0, 1, 2])

    manager = object.__new__(IOManager)
    manager.loader = Loader()
    manager.loader_iter = None
    manager._resume_skip_batches = 2

    manager.reset_loader()

    assert next(manager.loader_iter) == 2
    assert manager._resume_skip_batches == 0


def test_io_manager_warns_when_resume_must_replay_or_restart_workers():
    """Generic samplers and worker RNG should expose exact-resume limits."""
    manager = object.__new__(IOManager)
    manager.loader = SimpleNamespace(
        sampler=object(),
        batch_size=2,
        num_workers=2,
    )
    manager.loader_iter = None
    manager._resume_skip_batches = 0

    with pytest.warns(RuntimeWarning) as records:
        manager.restore_checkpoint_state(
            {"batch_offset": 2, "sampler": None, "num_workers": 2}
        )

    assert len(records) == 2
    assert manager._resume_skip_batches == 2


def test_io_manager_warns_when_resume_changes_batch_size():
    """Restored sample order cannot preserve old batch boundaries after resizing."""

    class Sampler:
        def __init__(self):
            self.offset = None

        def load_state_dict(self, _state, offset=0):
            self.offset = offset

    sampler = Sampler()
    manager = object.__new__(IOManager)
    manager.loader = SimpleNamespace(
        sampler=sampler,
        batch_size=4,
        num_workers=0,
    )
    manager.loader_iter = object()
    manager._resume_skip_batches = 0

    with pytest.warns(RuntimeWarning, match="batch size changed"):
        manager.restore_checkpoint_state(
            {
                "batch_offset": 1,
                "batch_size": 2,
                "sampler": {"indices": [4, 5]},
                "num_workers": 0,
            }
        )

    assert sampler.offset == 2


def test_io_manager_reports_composite_dataset_provenance():
    """Resolved provenance should preserve joint and mixed source topology."""

    class Dataset:
        name = "larcv"

        def __init__(self, files):
            self.reader = SimpleNamespace(file_paths=files)

        def __len__(self):
            return 2

    class Joint:
        name = "joint"
        joint = True

        def __init__(self):
            self.primary = Dataset(["primary.root"])
            self.secondary = Dataset(["secondary.root"])

        def __len__(self):
            return 2

    joint = Joint()
    manager = object.__new__(IOManager)
    manager.loader = SimpleNamespace(dataset=joint)
    manager.reader = joint.primary.reader

    provenance = manager.dataset_provenance()

    assert provenance["type"] == "joint"
    assert provenance["sources"]["primary"]["files"] == ["primary.root"]
    assert provenance["sources"]["secondary"]["files"] == ["secondary.root"]

    class Mixed:
        name = "mixed"

        def __init__(self):
            self.primary = Dataset(["primary.root"])
            self.cache = Dataset(["cache.h5"])

        def __len__(self):
            return 2

    mixed = IOManager._dataset_provenance(Mixed())
    assert mixed["sources"]["larcv"]["files"] == ["primary.root"]
    assert mixed["sources"]["hdf5"]["files"] == ["cache.h5"]

    manager.loader = None
    manager.reader = FakeReader()
    assert manager.dataset_provenance()["files"] == FakeReader.file_paths
    manager.reader = None
    assert manager.dataset_provenance() is None
