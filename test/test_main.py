"""Tests for the main SPINE runtime orchestration helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spine import main


def test_run_requires_base_block():
    """Top-level run should reject configs without a base section."""
    with pytest.raises(ValueError, match="base"):
        main.run({})


def test_run_dispatches_single_process(monkeypatch):
    """Non-distributed configs should dispatch through run_single."""
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(main, "process_world", lambda base: (False, 1, None))
    monkeypatch.setattr(main, "run_single", lambda cfg: calls.append(cfg))

    cfg = {"base": {}}
    main.run(cfg)

    assert calls == [cfg]


def test_run_dispatches_external_rank_training(monkeypatch):
    """Distributed runs with external rank should call train_single directly."""
    calls: list[tuple[int | None, bool, int | None, str | None]] = []

    monkeypatch.setattr(main, "process_world", lambda base: (True, 4, "file_system"))
    monkeypatch.setattr(
        main,
        "train_single",
        lambda rank, cfg, distributed, world_size, torch_sharing: calls.append(
            (rank, distributed, world_size, torch_sharing)
        ),
    )
    monkeypatch.setenv("RANK", "2")

    main.run({"base": {"train": {}}})

    assert calls == [(2, True, 4, "file_system")]

    with pytest.raises(ValueError, match="supported for training"):
        main.run({"base": {}})


def test_run_distributed_spawn_and_validation(monkeypatch):
    """Distributed runs should validate training mode and spawn when needed."""
    spawn_calls: list[tuple[object, tuple[object, ...], int]] = []
    torch_runtime = SimpleNamespace(
        multiprocessing=SimpleNamespace(
            spawn=lambda fn, args, nprocs: spawn_calls.append((fn, args, nprocs))
        )
    )

    monkeypatch.setattr(main, "process_world", lambda base: (True, 3, None))
    monkeypatch.setattr(main, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(main, "torch", torch_runtime)
    monkeypatch.delenv("RANK", raising=False)

    with pytest.raises(ValueError, match="training"):
        main.run({"base": {}})

    cfg = {"base": {"train": {}}}
    main.run(cfg)

    assert spawn_calls == [(main.train_single, (cfg, True, 3, None), 3)]


def test_run_single_dispatch(monkeypatch):
    """run_single should split between train and inference modes."""
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        main, "train_single", lambda **kwargs: calls.append(("train", kwargs["cfg"]))
    )
    monkeypatch.setattr(
        main, "inference_single", lambda cfg: calls.append(("infer", cfg))
    )

    train_cfg = {"base": {"train": {}}}
    infer_cfg = {"base": {}}
    main.run_single(train_cfg)
    main.run_single(infer_cfg)

    assert calls == [("train", train_cfg), ("infer", infer_cfg)]


def test_train_single_requires_torch(monkeypatch):
    """Training should fail fast when torch is unavailable."""
    monkeypatch.setattr(main, "TORCH_AVAILABLE", False)

    with pytest.raises(ImportError, match="PyTorch is required"):
        main.train_single(rank=None, cfg={"base": {"train": {}}})


def test_train_single_distributed_flow(monkeypatch):
    """Distributed training should set sharing, setup DDP, run, and tear down."""
    calls: list[tuple[str, object]] = []
    torch_runtime = SimpleNamespace(
        multiprocessing=SimpleNamespace(
            set_sharing_strategy=lambda strategy: calls.append(("sharing", strategy))
        ),
        distributed=SimpleNamespace(
            destroy_process_group=lambda: calls.append(("destroy", None))
        ),
    )

    class DummyDriver:
        def __init__(self, cfg, rank):
            calls.append(("driver", rank))

        def run(self):
            calls.append(("run", None))

    monkeypatch.setattr(main, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(main, "torch", torch_runtime)
    monkeypatch.setattr(
        main,
        "setup_ddp",
        lambda rank, world_size: calls.append(("setup", (rank, world_size))),
    )
    monkeypatch.setattr(main, "Driver", DummyDriver)

    main.train_single(
        rank=1,
        cfg={"base": {"train": {}}},
        distributed=True,
        world_size=4,
        torch_sharing="file_system",
    )

    assert calls == [
        ("sharing", "file_system"),
        ("setup", (1, 4)),
        ("driver", 1),
        ("run", None),
        ("destroy", None),
    ]


def test_inference_single_weight_handling(monkeypatch):
    """Inference should handle missing, scalar, and multiple weight paths."""
    calls: list[tuple[str, object]] = []

    class NoModelDriver:
        model = None

        def __init__(self, cfg):
            pass

        def run(self):
            calls.append(("run_none", None))

    monkeypatch.setattr(main, "Driver", NoModelDriver)
    main.inference_single({"base": {}})
    assert calls == [("run_none", None)]

    class SingleModel:
        weight_path = "weights.ckpt"

    class SingleDriver:
        model = SingleModel()

        def __init__(self, cfg):
            pass

        def initialize_log(self):
            calls.append(("log_single", None))

        def run(self):
            calls.append(("run_single", None))

    monkeypatch.setattr(main, "Driver", SingleDriver)
    main.inference_single({"base": {}})

    class MultiModel:
        weight_path = ["weights/b.ckpt", "weights/a.ckpt"]

        def load_weights(self, weight_path):
            calls.append(("load", weight_path))

    class MultiDriver:
        model = MultiModel()

        def __init__(self, cfg):
            pass

        def initialize_log(self):
            calls.append(("log_multi", None))

        def run(self):
            calls.append(("run_multi", None))

    infos: list[tuple[str, int, str]] = []
    monkeypatch.setattr(main, "Driver", MultiDriver)
    monkeypatch.setattr(
        main.logger,
        "info",
        lambda message, count, weights: infos.append((message, count, weights)),
    )
    main.inference_single({"base": {}})

    assert calls == [
        ("run_none", None),
        ("run_single", None),
        ("load", "weights/a.ckpt"),
        ("log_multi", None),
        ("run_multi", None),
        ("load", "weights/b.ckpt"),
        ("log_multi", None),
        ("run_multi", None),
    ]
    assert infos and infos[0][1] == 2


def test_process_world_and_setup_ddp(monkeypatch):
    """World parsing and DDP setup should validate and call torch correctly."""
    levels: list[str] = []
    monkeypatch.setattr(main.logger, "setLevel", lambda level: levels.append(level))
    monkeypatch.setattr(main, "set_visible_devices", lambda world_size, gpus: 2)

    with pytest.raises(ValueError, match="distributed execution is disabled"):
        main.process_world({"verbosity": "debug", "distributed": False})

    distributed, world_size, strategy = main.process_world(
        {
            "verbosity": "warning",
            "distributed": True,
            "torch_sharing_strategy": "file_system",
        }
    )
    assert levels == ["DEBUG", "WARNING"]
    assert (distributed, world_size, strategy) == (True, 2, "file_system")

    monkeypatch.setattr(main, "set_visible_devices", lambda world_size, gpus: 1)
    with pytest.raises(ValueError, match="torch_sharing_strategy"):
        main.process_world({"torch_sharing_strategy": "bad"})

    ddp_calls: list[tuple[str, object]] = []
    torch_runtime = SimpleNamespace(
        distributed=SimpleNamespace(
            init_process_group=lambda **kwargs: ddp_calls.append(("init", kwargs))
        ),
        cuda=SimpleNamespace(
            set_device=lambda device: ddp_calls.append(("device", device))
        ),
    )
    monkeypatch.setattr(main, "torch", torch_runtime)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "7")

    main.setup_ddp(rank=2, world_size=8, backend="gloo")

    assert ddp_calls == [
        ("init", {"backend": "gloo", "rank": 2, "world_size": 8}),
        ("device", 7),
    ]
    assert main.os.environ["MASTER_ADDR"] == "localhost"
    assert main.os.environ["MASTER_PORT"] == "12355"
