"""Tests for the central :mod:`spine.driver` orchestration class."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import spine.driver as driver_mod
import spine.io.manager as io_manager_mod
from spine.driver import Driver


class FakeTime:
    """Small stopwatch time payload."""

    wall = 2.0
    cpu = 1.0


class FakeWatch:
    """Stopwatch-like object used by Driver.log/process tests."""

    def __init__(self) -> None:
        self.running = False
        self.paused = False
        self.time = FakeTime()
        self.time_sum = FakeTime()


class FakeWatchManager:
    """Minimal StopwatchManager replacement with call recording."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.watches: dict[str, FakeWatch] = {}

    def initialize(self, key: str) -> None:
        self.calls.append(("initialize", key))
        self.watches.setdefault(key, FakeWatch())

    def start(self, key: str) -> None:
        self.calls.append(("start", key))
        self.watches.setdefault(key, FakeWatch()).running = True

    def stop(self, key: str) -> None:
        self.calls.append(("stop", key))
        self.watches.setdefault(key, FakeWatch()).running = False

    def update(self, watch: object, key: str | None = None) -> None:
        self.calls.append(("update", key))

    def reset(self) -> None:
        self.calls.append(("reset", None))
        for watch in self.watches.values():
            watch.running = False
            watch.paused = False

    def values(self):
        return self.watches.values()

    def items(self):
        return self.watches.items()

    def time(self, key: str) -> FakeTime:
        return self.watches[key].time


class FakeReader:
    """Reader-like object with entry and run/event accessors."""

    file_paths = ["/tmp/input_000.root", "/tmp/input_001.root"]
    cfg = {"post": {"calib": {}, "pid": {}}}

    def __init__(self, length: int = 3) -> None:
        self.length = length
        self.calls: list[tuple[str, object]] = []

    def __len__(self) -> int:
        return self.length

    def get(self, entry: int) -> dict[str, object]:
        self.calls.append(("get", entry))
        return {"index": entry}

    def get_run_event(self, run: int, subrun: int, event: int) -> dict[str, object]:
        self.calls.append(("get_run_event", (run, subrun, event)))
        return {"index": event}

    def process_entry_list(self, *args: object) -> None:
        self.calls.append(("process_entry_list", args))


class FakeLoader:
    """Loader-like iterable exposing a dataset reader."""

    def __init__(self) -> None:
        self.dataset = SimpleNamespace(reader=FakeReader(length=2))
        self.sampler = SimpleNamespace(epochs=[])
        self.batches = iter([{"index": [0, 1]}, {"index": [2, 3]}])

        def set_epoch(epoch: int) -> None:
            self.sampler.epochs.append(epoch)

        self.sampler.set_epoch = set_epoch

    def __len__(self) -> int:
        return 2

    def __iter__(self):
        self.batches = iter([{"index": [0, 1]}, {"index": [2, 3]}])
        return self

    def __next__(self) -> dict[str, object]:
        return next(self.batches)


def bare_driver() -> Driver:
    """Construct a Driver instance without running its initializer."""
    drv = object.__new__(Driver)
    drv.watch = FakeWatchManager()
    drv.watch.initialize("iteration")
    drv.tensorboard_cfg = None
    return drv


def test_driver_import_contract():
    """Driver should remain importable from the package root."""
    from spine import Driver as RootDriver
    from spine import __version__
    from spine.banner import ascii_logo

    assert RootDriver is Driver
    assert isinstance(__version__, str)
    assert "██████████" in ascii_logo
    assert "Central SPINE driver" in Driver.__doc__


def test_process_config_normalizes_and_logs(monkeypatch):
    """process_config should normalize seeds, sampler config and stored cfg."""
    drv = bare_driver()
    levels: list[str] = []
    infos: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        driver_mod.logger, "setLevel", lambda level: levels.append(level)
    )
    monkeypatch.setattr(driver_mod.logger, "info", lambda *args: infos.append(args))
    monkeypatch.setattr(driver_mod, "set_visible_devices", lambda world_size, gpus: 2)
    monkeypatch.setattr(driver_mod.time, "time", lambda: 123.4)
    monkeypatch.setattr(
        driver_mod.sc, "getstatusoutput", lambda cmd: (0, "test-kernel")
    )

    input_base = {"verbosity": "debug", "seed": -1, "gpus": [0, 1]}
    input_io = {"loader": {"sampler": "random"}}
    base, io, geo, model, build, post, ana = drv.process_config(
        io=input_io,
        base=input_base,
        geo={"detector": "icarus"},
        model={"name": "model"},
        build={"mode": "reco"},
        post={"module": {}},
        ana={"script": {}},
        rank=0,
    )

    assert input_base == {"verbosity": "debug", "seed": -1, "gpus": [0, 1]}
    assert input_io == {"loader": {"sampler": "random"}}
    assert levels == ["DEBUG"]
    assert base["world_size"] == 2
    assert base["seed"] == 123
    assert io["loader"]["sampler"] == {"name": "random", "seed": 123}
    assert drv.cfg == {
        "base": base,
        "io": io,
        "geo": geo,
        "model": model,
        "build": build,
        "post": post,
        "ana": ana,
    }
    assert any("test-kernel" in args for args in infos)
    assert not any("██████████" in str(args) for args in infos)


def test_process_config_validates_required_io_and_seed(monkeypatch):
    """process_config should reject missing I/O and non-integer seeds."""
    drv = bare_driver()
    monkeypatch.setattr(driver_mod, "set_visible_devices", lambda world_size, gpus: 0)

    base, *_ = drv.process_config(io={"reader": {}}, base=None, rank=1)
    assert base["world_size"] == 0
    assert isinstance(base["seed"], int)

    with pytest.raises(ValueError, match="io"):
        drv.process_config(io=None, base={})

    with pytest.raises(TypeError, match="integer"):
        drv.process_config(io={"reader": {}}, base={"seed": 1.5}, rank=1)

    with pytest.raises(TypeError, match="sampler"):
        drv.process_config(io={"loader": {"sampler": object()}}, base={}, rank=1)


def test_process_config_reuses_one_generated_seed(monkeypatch):
    """process_config should reuse one generated default seed per config pass."""
    drv = bare_driver()

    time_values = iter([123.4, 987.6])
    monkeypatch.setattr(driver_mod.time, "time", lambda: next(time_values))
    monkeypatch.setattr(driver_mod, "set_visible_devices", lambda world_size, gpus: 0)
    monkeypatch.setattr(driver_mod.logger, "setLevel", lambda level: None)
    monkeypatch.setattr(driver_mod.logger, "info", lambda *args: None)
    monkeypatch.setattr(
        driver_mod.sc, "getstatusoutput", lambda cmd: (0, "test-kernel")
    )

    base, io, *_ = drv.process_config(
        io={"loader": {"sampler": {"name": "random"}}},
        base={},
        rank=0,
    )

    assert base["seed"] == 123
    assert io["loader"]["sampler"]["seed"] == 123


def test_initialize_base_sets_runtime_state(monkeypatch):
    """initialize_base should seed RNGs and derive rank/distributed state."""
    drv = bare_driver()
    seeds: list[tuple[str, int]] = []

    monkeypatch.setattr(
        driver_mod.random, "seed", lambda seed: seeds.append(("py", seed))
    )
    monkeypatch.setattr(
        driver_mod.np.random, "seed", lambda seed: seeds.append(("np", seed))
    )
    monkeypatch.setattr(
        driver_mod, "numba_seed", lambda seed: seeds.append(("nb", seed))
    )
    monkeypatch.setattr(
        driver_mod.runtime, "manual_seed", lambda seed: seeds.append(("torch", seed))
    )

    train = drv.initialize_base(
        seed=7,
        world_size=2,
        rank=1,
        distributed=False,
        dtype="float64",
        log_dir="out",
        prefix_log=True,
        overwrite_log=True,
        iterations=5,
        unwrap=True,
        split_output=True,
        train={"optimizer": {}},
    )

    assert train == {"optimizer": {}}
    assert seeds == [("py", 7), ("np", 7), ("nb", 7), ("torch", 7)]
    assert drv.rank == 1
    assert drv.main_process is False
    assert drv.distributed is True
    assert drv.dtype == "float64"
    assert drv.iterations == 5
    assert drv.unwrap is True
    assert drv.split_output is True

    drv.initialize_base(seed=3, world_size=1, rank=None)
    assert drv.rank == 0
    assert drv.main_process is True

    with pytest.raises(ValueError, match="without specifying"):
        drv.initialize_base(seed=1, world_size=2, rank=None)

    with pytest.raises(ValueError, match="rank index"):
        drv.initialize_base(seed=1, world_size=1, rank=2)


def test_extract_driver_base_config_filters_runtime_keys():
    """extract_driver_base_config should keep launcher-only keys out of initialize_base."""
    base = {
        "seed": 7,
        "dtype": "float32",
        "world_size": 2,
        "gpus": [0, 1],
        "verbosity": "debug",
        "torch_sharing_strategy": "file_system",
        "log_dir": "logs",
        "distributed": True,
        "tensorboard": True,
    }

    driver_base = Driver.extract_driver_base_config(base)

    assert driver_base == {
        "seed": 7,
        "dtype": "float32",
        "world_size": 2,
        "log_dir": "logs",
        "distributed": True,
        "tensorboard": True,
    }


def test_extract_driver_base_config_rejects_unknown_keys():
    """extract_driver_base_config should fail on unrecognized base settings."""
    with pytest.raises(KeyError, match="unexpected_key"):
        Driver.extract_driver_base_config({"seed": 7, "unexpected_key": 1})


def test_initialize_io_reader_writer_and_iteration_harmonization(monkeypatch):
    """Reader I/O initialization should set prefixes, writer and iteration count."""
    drv = bare_driver()
    drv.cfg = {"base": {}, "io": {}}
    drv.rank = None
    drv.dtype = "float32"
    drv.world_size = 0
    drv.distributed = False
    drv.unwrap = False
    drv.iterations = -1
    drv.epochs = None
    drv.split_output = False
    reader = FakeReader(length=4)
    writer_calls: list[tuple[object, str, bool]] = []

    monkeypatch.setattr(io_manager_mod, "reader_factory", lambda cfg: reader)
    monkeypatch.setattr(
        io_manager_mod,
        "writer_factory",
        lambda cfg, prefix, split: writer_calls.append((cfg, prefix, split))
        or "writer",
    )

    drv.initialize_io({"reader": {"name": "hdf5"}, "writer": {"name": "csv"}})

    assert drv.io.reader is reader
    assert drv.io.iter_per_epoch == 4
    assert drv.io.post_list == ("calib", "pid")
    assert drv.iterations == 4
    assert drv.epochs == 1.0
    assert drv.io.writer == "writer"
    assert writer_calls == [({"name": "csv"}, "input_000--input_001", False)]

    drv = bare_driver()
    drv.cfg = {"base": {}, "io": {}}
    drv.rank = None
    drv.dtype = "float32"
    drv.world_size = 0
    drv.distributed = False
    drv.unwrap = False
    drv.iterations = None
    drv.epochs = None
    drv.split_output = False

    drv.initialize_io({"reader": {"name": "hdf5"}})
    assert drv.iterations is None
    assert drv.epochs is None

    with pytest.raises(ValueError, match="either a loader or a reader"):
        drv.initialize_io({})
    with pytest.raises(ValueError, match="either a loader or a reader"):
        drv.initialize_io({"loader": {}, "reader": {}})


def test_initialize_io_loader_paths(monkeypatch):
    """Loader I/O initialization should require torch and optionally unwrap."""
    drv = bare_driver()
    drv.cfg = {}
    drv.rank = 0
    drv.dtype = "float32"
    drv.world_size = 2
    drv.distributed = True
    drv.unwrap = True
    drv.iterations = None
    drv.epochs = 1.5
    drv.split_output = True
    loader = FakeLoader()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(io_manager_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        io_manager_mod,
        "loader_factory",
        lambda **kwargs: calls.append(kwargs) or loader,
    )
    monkeypatch.setattr(io_manager_mod, "Unwrapper", lambda: "unwrapper")

    drv.initialize_io({"loader": {"batch_size": 2}})

    assert drv.io.loader is loader
    assert drv.io.reader is loader.dataset.reader
    assert drv.io.unwrapper == "unwrapper"
    assert drv.io.post_list == ()
    assert drv.iterations == 3
    assert drv.io.output_prefix == ["input_000", "input_001"]
    assert calls[0]["distributed"] is True

    drv = bare_driver()
    drv.cfg = {}
    drv.rank = None
    drv.dtype = "float32"
    drv.world_size = 0
    drv.distributed = False
    drv.unwrap = False
    drv.iterations = None
    drv.epochs = 1.0
    drv.split_output = False
    monkeypatch.setattr(io_manager_mod, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError, match="loader"):
        drv.initialize_io({"loader": {}})


def test_initialize_log_names_and_writer(monkeypatch, tmp_path):
    """initialize_log should choose generic or model-specific log names."""
    created: list[tuple[str, bool, int, object, str]] = []

    class DummyLogManager:
        def __init__(
            self,
            path,
            overwrite=False,
            buffer_size=1,
            tensorboard=None,
            tensorboard_dir=None,
        ):
            created.append((path, overwrite, buffer_size, tensorboard, tensorboard_dir))

    monkeypatch.setattr(driver_mod, "LogManager", DummyLogManager)

    drv = bare_driver()
    drv.log_dir = str(tmp_path / "logs")
    drv.builder = object()
    drv.model = None
    drv.prefix_log = True
    drv.overwrite_log = True
    drv.csv_buffer_size = 10
    drv.io = SimpleNamespace(
        format_log_name=lambda log_name, log_dir: f"sample_{log_name}",
    )
    drv.initialize_log()
    assert created[-1][0].endswith("sample_spine_log.csv")

    drv.io = SimpleNamespace(
        format_log_name=lambda log_name, log_dir: f"{'x' * 26}_{log_name}",
    )
    drv.initialize_log()
    assert os.path.basename(created[-1][0]) == f"{'x' * 26}_spine_log.csv"

    drv.builder = None
    drv.model = SimpleNamespace(start_iteration=12, train=True, distributed=True)
    drv.rank = 3
    drv.prefix_log = False
    drv.initialize_log()
    assert created[-1][0].endswith("train_proc3_log-0000012.csv")

    drv.model = SimpleNamespace(start_iteration=5, train=False, distributed=False)
    drv.initialize_log()
    assert created[-1][0].endswith("inference_log-0000005.csv")


def test_initialize_tensorboard_logger(monkeypatch, tmp_path):
    """initialize_log should optionally create a TensorBoard writer."""
    created: list[tuple[str, object, str]] = []

    class DummyLogManager:
        def __init__(
            self,
            path,
            overwrite=False,
            buffer_size=1,
            tensorboard=None,
            tensorboard_dir=None,
        ):
            created.append((path, tensorboard, tensorboard_dir))

    monkeypatch.setattr(driver_mod, "LogManager", DummyLogManager)

    drv = bare_driver()
    drv.log_dir = str(tmp_path / "logs")
    drv.builder = object()
    drv.model = None
    drv.prefix_log = False
    drv.overwrite_log = False
    drv.csv_buffer_size = 1
    drv.tensorboard_cfg = {"log_dir": "tb", "flush_secs": 5}
    drv.initialize_log()

    assert created[-1] == (
        os.path.join(drv.log_dir, "spine_log.csv"),
        {"log_dir": "tb", "flush_secs": 5},
        os.path.join(drv.log_dir, "tensorboard"),
    )

    drv.tensorboard_cfg = True
    drv.initialize_log()
    assert created[-1] == (
        os.path.join(drv.log_dir, "spine_log.csv"),
        True,
        os.path.join(drv.log_dir, "tensorboard"),
    )


def test_driver_constructor_initializes_optional_managers(monkeypatch):
    """__init__ should connect optional geometry/model/build/post/ana managers."""
    calls: list[tuple[str, object]] = []

    def process_config(self, **kwargs):
        calls.append(("process_config", kwargs["rank"]))
        return (
            {"seed": 1},
            {},
            {"detector": "dummy"},
            {"model": "cfg"},
            {"build": "cfg"},
            {"post": "cfg"},
            {"ana": "cfg"},
        )

    def initialize_base(self, **kwargs):
        calls.append(("initialize_base", kwargs["rank"]))
        self.dtype = "float32"
        self.rank = kwargs["rank"]
        self.distributed = False
        self.unwrap = True
        self.parent_path = "/tmp"
        self.log_dir = "logs"
        return {"train": "cfg"}

    def initialize_io(self, io):
        calls.append(("initialize_io", io))
        self.io = SimpleNamespace(
            has_loader=True,
            iter_per_epoch=2,
            post_list=("previous",),
            log_prefix="input",
        )

    class FakeModel:
        to_numpy = True

        def __init__(self, **kwargs):
            calls.append(("model", kwargs))

    monkeypatch.setattr(driver_mod, "StopwatchManager", FakeWatchManager)
    monkeypatch.setattr(Driver, "process_config", process_config)
    monkeypatch.setattr(Driver, "initialize_base", initialize_base)
    monkeypatch.setattr(Driver, "initialize_io", initialize_io)
    monkeypatch.setattr(
        driver_mod.GeoManager,
        "initialize_or_get",
        lambda **kwargs: calls.append(("geo", kwargs)),
    )
    monkeypatch.setattr(driver_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(driver_mod, "ModelManager", FakeModel)
    monkeypatch.setattr(
        driver_mod,
        "BuildManager",
        lambda **kwargs: calls.append(("build", kwargs)) or "builder",
    )
    monkeypatch.setattr(
        driver_mod,
        "PostManager",
        lambda post, post_list, parent_path: calls.append(
            ("post", (post, post_list, parent_path))
        )
        or "post",
    )
    monkeypatch.setattr(
        driver_mod,
        "AnaManager",
        lambda ana, log_dir, prefix: calls.append(("ana", (ana, log_dir, prefix)))
        or "ana",
    )

    drv = Driver({"base": {}}, rank=0)

    assert drv.model is not None
    assert drv.builder == "builder"
    assert drv.post == "post"
    assert drv.ana == "ana"
    assert ("geo", {"detector": "dummy"}) in calls
    assert any(call[0] == "model" for call in calls)
    assert any(call[0] == "build" for call in calls)
    assert any(call[0] == "post" for call in calls)
    assert any(call[0] == "ana" for call in calls)


def test_driver_constructor_validates_model_dependent_modes(monkeypatch):
    """__init__ should reject inconsistent train/model/build/post/ana requests."""

    def configure(
        process_return,
        initialize_base_return=None,
        loader=None,
        unwrap=False,
    ):
        monkeypatch.setattr(driver_mod, "StopwatchManager", FakeWatchManager)
        monkeypatch.setattr(
            Driver,
            "process_config",
            lambda self, **kwargs: process_return,
        )

        def initialize_base(self, **kwargs):
            self.dtype = "float32"
            self.rank = None
            self.distributed = False
            self.unwrap = unwrap
            self.parent_path = None
            self.log_dir = "logs"
            return initialize_base_return

        def initialize_io(self, io):
            self.io = SimpleNamespace(
                has_loader=loader is not None,
                iter_per_epoch=1,
                post_list=(),
                log_prefix="input",
            )

        monkeypatch.setattr(Driver, "initialize_base", initialize_base)
        monkeypatch.setattr(Driver, "initialize_io", initialize_io)

    configure(({}, {}, None, None, None, None, None), initialize_base_return={})
    with pytest.raises(ValueError, match="no model"):
        Driver({})

    configure(({}, {}, None, {"model": "cfg"}, None, None, None), loader=None)
    with pytest.raises(ValueError, match="loader"):
        Driver({})

    monkeypatch.setattr(driver_mod, "TORCH_AVAILABLE", False)
    configure(({}, {}, None, {"model": "cfg"}, None, None, None), loader=object())
    with pytest.raises(ImportError, match="model functionality"):
        Driver({})

    class NumpyModel:
        to_numpy = True

        def __init__(self, **kwargs):
            pass

    class TensorModel:
        to_numpy = False

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(driver_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(driver_mod, "ModelManager", NumpyModel)
    configure(
        ({}, {}, None, {"model": "cfg"}, {"build": "cfg"}, None, None), loader=object()
    )
    with pytest.raises(ValueError, match="build representations"):
        Driver({})

    monkeypatch.setattr(driver_mod, "ModelManager", TensorModel)
    configure(
        ({}, {}, None, {"model": "cfg"}, {"build": "cfg"}, None, None),
        loader=object(),
        unwrap=True,
    )
    with pytest.raises(ValueError, match="numpy"):
        Driver({})

    monkeypatch.setattr(driver_mod, "ModelManager", NumpyModel)
    configure(
        ({}, {}, None, {"model": "cfg"}, None, {"post": "cfg"}, None), loader=object()
    )
    with pytest.raises(ValueError, match="post-processors"):
        Driver({})

    configure(
        ({}, {}, None, {"model": "cfg"}, None, None, {"ana": "cfg"}), loader=object()
    )
    with pytest.raises(ValueError, match="analysis scripts"):
        Driver({})


def test_optional_initializers_accept_absent_configs():
    """Optional driver managers should stay unset when configs are omitted."""
    drv = bare_driver()
    drv.io = SimpleNamespace(has_loader=False, post_list=None, log_prefix="input")
    drv.initialize_model(None, None)
    drv.initialize_ana(None)

    assert drv.model is None
    assert drv.ana is None


def test_initialize_io_validation_branches(monkeypatch):
    """initialize_io should reject incompatible writer and iteration settings."""
    drv = bare_driver()
    drv.cfg = {}
    drv.rank = None
    drv.dtype = "float32"
    drv.world_size = 0
    drv.distributed = False
    drv.unwrap = False
    drv.iterations = None
    drv.epochs = 1.0
    drv.split_output = False

    monkeypatch.setattr(io_manager_mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(io_manager_mod, "loader_factory", lambda **kwargs: FakeLoader())
    with pytest.raises(ValueError, match="write"):
        drv.initialize_io({"loader": {"name": "loader"}, "writer": {"name": "writer"}})

    drv = bare_driver()
    drv.cfg = {}
    drv.rank = None
    drv.dtype = "float32"
    drv.world_size = 0
    drv.distributed = False
    drv.unwrap = False
    drv.iterations = 1
    drv.epochs = 1
    drv.split_output = False

    monkeypatch.setattr(
        io_manager_mod, "reader_factory", lambda cfg: FakeReader(length=2)
    )
    with pytest.raises(ValueError, match="iterations"):
        drv.initialize_io({"reader": {"name": "reader"}})


def test_iteration_and_load_paths():
    """Driver iteration should dispatch through process correctly."""

    class IterIO(SimpleNamespace):
        def __len__(self):
            return 2

    drv = bare_driver()
    drv.io = IterIO(
        has_loader=False,
        reset_loader=lambda: None,
        close=lambda: None,
    )
    processed: list[object] = []
    drv.process = lambda entry=None, **kwargs: processed.append(entry) or {
        "index": entry
    }

    assert iter(drv) is drv
    assert next(drv) == {"index": 0}
    assert next(drv) == {"index": 1}
    with pytest.raises(StopIteration):
        next(drv)

    drv = bare_driver()
    drv.io = IterIO(
        has_loader=True,
        reset_loader=lambda: None,
        close=lambda: None,
    )
    processed.clear()
    drv.process = lambda entry=None, **kwargs: processed.append(entry) or {
        "index": [0, 1]
    }
    assert iter(drv) is drv
    assert next(drv) == {"index": [0, 1]}
    assert processed == [None]


def test_process_runs_pipeline_in_order():
    """process should call each optional processing stage in order."""
    drv = bare_driver()
    drv.watch.watches["iteration"].running = True
    calls: list[str] = []
    drv.io = SimpleNamespace(
        load=lambda *args: calls.append("load") or {"index": 0},
        unwrap=lambda data: calls.append("unwrap") or data,
        write=lambda data, cfg: calls.append("write"),
        watch="io-watch",
    )

    class Model:
        watch = "model-watch"

        def __call__(self, data, iteration=None, epoch=None):
            calls.append(f"model:{iteration}:{epoch}")
            return {"loss": 1.0}

    drv.model = Model()
    drv.builder = lambda data: calls.append("build")
    drv.post = SimpleNamespace(
        watch="post-watch", __call__=lambda data: calls.append("post")
    )
    drv.post = type(
        "Post",
        (),
        {"watch": "post-watch", "__call__": lambda self, data: calls.append("post")},
    )()
    drv.ana = type(
        "Ana",
        (),
        {"watch": "ana-watch", "__call__": lambda self, data: calls.append("ana")},
    )()
    drv.cfg = {"base": {}}

    data = drv.process(entry=4, iteration=9, epoch=1.5)

    assert data == {"index": 0, "loss": 1.0}
    assert calls == ["load", "model:9:1.5", "unwrap", "build", "post", "ana", "write"]
    assert ("reset", None) in drv.watch.calls
    assert ("update", None) in drv.watch.calls
    assert drv.watch.calls[-1] == ("stop", "iteration")


def test_run_loop_resets_loader_logs_and_closes():
    """run should iterate requested entries and close output resources."""
    drv = bare_driver()
    drv.iterations = 3
    drv.model = SimpleNamespace(train=True, start_iteration=1)
    drv.distributed = True
    drv.ana = SimpleNamespace(close=lambda: calls.append("ana_close"))
    drv.log_manager = SimpleNamespace(close=lambda: calls.append("log_close"))
    calls: list[object] = []
    sampler_epochs: list[int] = []
    drv.io = SimpleNamespace(
        has_loader=True,
        iter_per_epoch=2,
        prepare_iteration=lambda iteration: (
            sampler_epochs.append(iteration // 2) if iteration % 2 == 0 else None
        ),
        close=lambda: calls.append("io_close"),
    )
    drv.initialize_log = lambda: calls.append("initialize_log")
    drv.process = lambda entry=None, iteration=None, epoch=None: calls.append(
        ("process", entry, iteration, epoch)
    ) or {"index": iteration}
    drv.log = lambda data, tstamp, iteration, epoch: calls.append(
        ("log", data["index"], iteration, epoch)
    )

    drv.run()

    assert calls[0] == "initialize_log"
    assert ("process", None, 1, 1.0) in calls
    assert ("process", None, 2, 1.5) in calls
    assert sampler_epochs == [1]
    assert calls[-3:] == [
        "ana_close",
        "log_close",
        "io_close",
    ]

    drv.iterations = None
    with pytest.raises(ValueError, match="iterations"):
        drv.run()


def test_run_loop_cleans_up_on_processing_failure():
    """run should close resources even if processing raises."""
    drv = bare_driver()
    drv.iterations = 1
    drv.model = None
    drv.ana = SimpleNamespace(close=lambda: calls.append("ana_close"))
    drv.log_manager = SimpleNamespace(close=lambda: calls.append("log_close"))
    drv.io = SimpleNamespace(
        has_loader=False,
        iter_per_epoch=1,
        prepare_iteration=lambda iteration: calls.append("prepare"),
        close=lambda: calls.append("io_close"),
    )
    calls: list[str] = []
    drv.initialize_log = lambda: calls.append("initialize_log")

    def fail_process(**kwargs):
        calls.append("process")
        raise RuntimeError("boom")

    drv.process = fail_process

    with pytest.raises(RuntimeError, match="boom"):
        drv.run()

    assert calls == [
        "initialize_log",
        "prepare",
        "process",
        "ana_close",
        "log_close",
        "io_close",
    ]


def test_apply_filter_resets_loader_iterator():
    """apply_filter should delegate to reader and invalidate loader iterator."""
    drv = bare_driver()
    calls: list[tuple[object, ...]] = []
    drv.io = SimpleNamespace(
        apply_filter=lambda *args: calls.append(args),
    )

    drv.apply_filter(1, 2, [3], [4], [(1, 2, 3)], [(4, 5, 6)])

    assert calls == [(1, 2, [3], [4], [(1, 2, 3)], [(4, 5, 6)])]


def test_log_collects_scalars_memory_and_stdout(monkeypatch):
    """log should append scalar metrics and emit formatted progress lines."""
    drv = bare_driver()
    drv.watch.initialize("model")
    drv.log_step = 1
    drv.model = SimpleNamespace(train=True)
    drv.rank = 0
    drv.distributed = True
    drv.main_process = True
    rows: list[dict[str, object]] = []
    infos: list[str] = []
    drv.log_manager = SimpleNamespace(
        append=lambda data, watch, iteration, epoch: rows.append(
            {
                "first_entry": 7,
                "cpu_mem": 4.0,
                "gpu_mem": 2.0,
                "gpu_mem_perc": 25.0,
                "tensor_metric": 3.5,
            }
        )
        or rows[-1],
        log_stdout_summary=lambda *args, **kwargs: infos.append("Iter. 0"),
    )

    drv.log(
        {
            "index": [7, 8],
            "loss": 1.25,
            "accuracy": 0.5,
            "tensor_metric": 3.5,
        },
        "2026-01-01 00:00:00",
        iteration=0,
        epoch=0.5,
    )

    row = rows[-1]
    assert row["first_entry"] == 7
    assert row["cpu_mem"] == 4.0
    assert row["gpu_mem"] == 2.0
    assert row["gpu_mem_perc"] == 25.0
    assert row["tensor_metric"] == 3.5
    assert any("Iter. 0" in msg for msg in infos)

    drv.rank = None
    drv.distributed = False
    drv.main_process = False
    drv.model = None
    drv.log({"index": 9}, "2026-01-01 00:00:01", iteration=1, epoch=1.0)
    assert rows[-1]["gpu_mem"] == 2.0


def test_log_requires_initialized_log_manager():
    """log should fail clearly before initialize_log has been called."""
    drv = bare_driver()
    drv.log_manager = None

    with pytest.raises(RuntimeError, match="log manager"):
        drv.log({"index": 0}, "2026-01-01 00:00:00", iteration=0)
