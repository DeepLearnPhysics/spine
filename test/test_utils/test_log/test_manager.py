"""Tests for structured scalar logging utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import spine.utils.log.manager as log_manager_mod
from spine.utils.log import LogManager
from spine.utils.log.manager import get_first_entry


class FakeTime:
    """Small stopwatch time payload."""

    wall = 2.0
    cpu = 1.0


class FakeWatch:
    """Stopwatch-like object."""

    time = FakeTime()
    time_sum = FakeTime()


class FakeWatchManager:
    """Minimal stopwatch manager."""

    def __init__(self) -> None:
        self.watches = {"iteration": FakeWatch(), "model": FakeWatch()}

    def items(self):
        return self.watches.items()

    def time(self, key: str) -> FakeTime:
        return self.watches[key].time


class FakeCSVWriter:
    """CSV writer stand-in."""

    def __init__(self, file_name, overwrite=False, buffer_size=1):
        self.file_name = file_name
        self.overwrite = overwrite
        self.buffer_size = buffer_size
        self.rows = []
        self.closed = False

    def append(self, row):
        self.rows.append(row)

    def close(self):
        self.closed = True


def test_log_manager_collects_and_writes_scalars(monkeypatch, tmp_path):
    """LogManager should collect memory, timing, scalar and tensor metrics."""
    writers: list[FakeCSVWriter] = []
    tb_scalars: list[tuple[str, float, int]] = []

    monkeypatch.setattr(
        log_manager_mod,
        "CSVWriter",
        lambda *args, **kwargs: writers.append(FakeCSVWriter(*args, **kwargs))
        or writers[-1],
    )
    monkeypatch.setattr(
        log_manager_mod.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(used=4.0e9, percent=50.0),
    )
    monkeypatch.setattr(log_manager_mod.runtime, "cuda_is_available", lambda: True)
    monkeypatch.setattr(log_manager_mod.runtime, "cuda_mem_info", lambda: (0, 8.0e9))
    monkeypatch.setattr(
        log_manager_mod.runtime, "cuda_max_memory_allocated", lambda: 2.0e9
    )
    monkeypatch.setattr(
        log_manager_mod.runtime, "create_summary_writer", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        log_manager_mod.runtime, "is_tensor", lambda value: hasattr(value, "dim")
    )

    class ScalarTensor:
        def dim(self):
            return 0

        def item(self):
            return 3.5

    manager = LogManager(
        str(tmp_path / "log.csv"),
        overwrite=True,
        buffer_size=10,
    )
    manager.tb_logger = SimpleNamespace(
        add_scalar=lambda key, value, step: tb_scalars.append((key, value, step)),
        flush=lambda: tb_scalars.append(("flush", 0, 0)),
        close=lambda: tb_scalars.append(("close", 0, 0)),
    )
    row = manager.append(
        {
            "index": np.array([7, 8]),
            "loss": 1.25,
            "accuracy": 0.5,
            "tensor_metric": ScalarTensor(),
        },
        FakeWatchManager(),
        iteration=0,
        epoch=0.5,
    )

    assert writers[0].overwrite is True
    assert writers[0].buffer_size == 10
    assert writers[0].rows == [row]
    assert row["first_entry"] == 7
    assert row["cpu_mem"] == 4.0
    assert row["gpu_mem"] == 2.0
    assert row["gpu_mem_perc"] == 25.0
    assert row["iteration_time"] == 2.0
    assert row["tensor_metric"] == 3.5
    assert ("loss", 1.25, 0) in tb_scalars
    assert ("tensor_metric", 3.5, 0) in tb_scalars

    manager.append_tensorboard({"iter": 1, "flag": True}, iteration=1)
    assert ("flag", 1, 1) in tb_scalars
    manager.close()
    assert writers[0].closed is True


def test_log_manager_tensorboard_paths(monkeypatch, tmp_path):
    """TensorBoard config should resolve default and relative directories."""
    tb_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(log_manager_mod, "CSVWriter", FakeCSVWriter)
    monkeypatch.setattr(
        log_manager_mod.runtime,
        "create_summary_writer",
        lambda log_dir, **kwargs: tb_calls.append((log_dir, kwargs)) or object(),
    )

    LogManager(
        str(tmp_path / "log.csv"),
        tensorboard=True,
        tensorboard_dir=str(tmp_path / "tensorboard"),
    )
    assert tb_calls[-1] == (str(tmp_path / "tensorboard"), {})

    LogManager(
        "log.csv",
        tensorboard=True,
        tensorboard_dir="logs/tensorboard",
    )
    assert tb_calls[-1] == ("logs/tensorboard", {})

    LogManager(
        str(tmp_path / "log.csv"),
        tensorboard={"log_dir": "tb", "flush_secs": 5},
        tensorboard_dir=str(tmp_path / "tensorboard"),
    )
    assert tb_calls[-1] == (str(tmp_path / "tb"), {"flush_secs": 5})

    with pytest.raises(ValueError, match="directory"):
        LogManager(str(tmp_path / "log.csv"), tensorboard=True)


def test_log_manager_stdout_summary(monkeypatch):
    """LogManager should emit the formatted progress table."""
    infos: list[str] = []
    barriers: list[None] = []
    monkeypatch.setattr(
        log_manager_mod.logger,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(
        log_manager_mod.runtime, "distributed_barrier", lambda: barriers.append(None)
    )
    monkeypatch.setattr(
        log_manager_mod.runtime,
        "distributed_all_gather_object",
        lambda obj: [
            obj,
            (
                1,
                "  | 1    | 1.50 s (90.00 %)    | 1.00 GB (10.00 %)    | 1.000    | 0.250    |",
            ),
        ],
    )

    LogManager.log_stdout_summary(
        {
            "cpu_mem": 4.0,
            "cpu_mem_perc": 50.0,
            "gpu_mem": 2.0,
            "gpu_mem_perc": 25.0,
            "model_time": 1.0,
        },
        {"loss": 1.25, "accuracy": 0.5},
        FakeWatchManager(),
        "2026-01-01 00:00:00",
        iteration=0,
        epoch=0.5,
        model_train=True,
        rank=0,
        distributed=True,
        main_process=True,
    )

    assert len(barriers) == 1
    assert any("Iter. 0" in msg for msg in infos)
    assert any("train" in msg for msg in infos)
    assert any("| 0    |" in msg for msg in infos)
    assert any("| 1    |" in msg for msg in infos)


def test_log_manager_optional_branches(monkeypatch, tmp_path):
    """Optional branches should be explicit and stable."""
    monkeypatch.setattr(log_manager_mod, "CSVWriter", FakeCSVWriter)
    manager = LogManager(str(tmp_path / "log.csv"))

    manager.append_tensorboard({"loss": 1.0}, iteration=0)

    infos: list[str] = []
    monkeypatch.setattr(
        log_manager_mod.logger,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    watch = FakeWatchManager()
    watch.watches["iteration"].time.wall = 0.0
    LogManager.log_stdout_summary(
        {
            "cpu_mem": 4.0,
            "cpu_mem_perc": 50.0,
            "gpu_mem": 0.0,
            "gpu_mem_perc": 0.0,
        },
        {},
        watch,
        "2026-01-01 00:00:00",
        iteration=1,
        epoch=None,
        model_train=False,
        rank=None,
        distributed=False,
        main_process=False,
    )

    assert infos


def test_log_manager_stdout_summary_non_main_rank_only_contributes_row(monkeypatch):
    """Non-main distributed ranks should not print headers directly."""
    infos: list[str] = []
    monkeypatch.setattr(
        log_manager_mod.logger,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(log_manager_mod.runtime, "distributed_barrier", lambda: None)
    monkeypatch.setattr(
        log_manager_mod.runtime,
        "distributed_all_gather_object",
        lambda obj: [obj],
    )

    LogManager.log_stdout_summary(
        {
            "cpu_mem": 4.0,
            "cpu_mem_perc": 50.0,
            "gpu_mem": 2.0,
            "gpu_mem_perc": 25.0,
            "model_time": 1.0,
        },
        {"loss": 1.25, "accuracy": 0.5},
        FakeWatchManager(),
        "2026-01-01 00:00:00",
        iteration=0,
        epoch=0.5,
        model_train=True,
        rank=1,
        distributed=True,
        main_process=False,
    )

    assert infos == []
    assert get_first_entry([1, 2]) == 1
    assert get_first_entry((3, 4)) == 3
    assert get_first_entry(np.array([5, 6])) == 5
    assert get_first_entry(7) == 7
