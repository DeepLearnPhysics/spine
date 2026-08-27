"""Tests for checkpoint-bound validation orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import spine.model.validation as validation_mod
from spine.model import ValidationManager
from spine.model.validation import BestCheckpoint, EarlyStopping
from spine.utils.conditional import TORCH_AVAILABLE
from spine.utils.stopwatch import StopwatchManager


def ordinary_loader() -> dict:
    """Return a representative ordinary training loader configuration."""
    return {
        "minibatch_size": 4,
        "shuffle": True,
        "num_workers": 2,
        "entry_list": [1, 2],
        "sampler": {"name": "random_sequence", "seed": 8},
        "dataset": {
            "name": "larcv",
            "file_keys": "train.root",
            "n_entry": 10,
            "augment": {"name": "translate"},
            "schema": {"data": {"parser": "sparse3d"}},
        },
    }


def test_build_loader_config_replaces_ordinary_source():
    """Ordinary validation should inherit schema but not training randomness."""
    loader = ordinary_loader()
    derived = ValidationManager.build_loader_config(
        loader, {"file_keys": "validation.root"}, seed=3
    )

    assert loader["dataset"]["file_keys"] == "train.root"
    assert derived["dataset"]["file_keys"] == "validation.root"
    assert derived["dataset"]["schema"] == loader["dataset"]["schema"]
    assert "augment" not in derived["dataset"]
    assert "n_entry" not in derived["dataset"]
    assert "entry_list" not in derived
    assert "sampler" not in derived
    assert derived["shuffle"] is False


def test_build_loader_config_preserves_joint_overlay_structure():
    """Joint validation should replace both sources and retain overlay rate."""
    loader = ordinary_loader()
    loader["sampler"] = {
        "name": "joint_random_sequence",
        "seed": 7,
        "pair_probability": 0.4,
    }
    loader["dataset"] = {
        "name": "joint",
        "base": {
            "name": "larcv",
            "schema": {"data": {"parser": "sparse3d"}},
            "file_keys": "inherited.root",
        },
        "primary": {"file_keys": "primary_train.root"},
        "secondary": {"file_keys": "secondary_train.root"},
        "augment": {"name": "translate"},
    }
    cfg = {
        "sources": {
            "primary": {"file_keys": "primary_val.root"},
            "secondary": {"file_list": "secondary_val.txt"},
        }
    }

    derived = ValidationManager.build_loader_config(loader, cfg, seed=11)

    assert "file_keys" not in derived["dataset"]["base"]
    assert derived["dataset"]["primary"] == {"file_keys": "primary_val.root"}
    assert derived["dataset"]["secondary"] == {"file_list": "secondary_val.txt"}
    assert "augment" not in derived["dataset"]
    assert derived["sampler"] == {
        "name": "joint_sequential",
        "seed": 11,
        "pair_probability": 0.4,
    }


def test_build_loader_config_replaces_aligned_mixed_sources():
    """Mixed validation should retain alignment settings and replace both inputs."""
    loader = ordinary_loader()
    loader["dataset"] = {
        "name": "mixed",
        "larcv": {"file_keys": "train.root", "schema": {"x": {}}},
        "hdf5": {"file_keys": "train.h5", "keys": ["prediction"]},
        "hdf5_key_map": {"old": "new"},
    }
    cfg = {
        "sources": {
            "larcv": {"file_keys": "validation.root"},
            "hdf5": {"file_keys": "validation.h5"},
        }
    }

    derived = ValidationManager.build_loader_config(loader, cfg, seed=2)

    assert derived["dataset"]["larcv"]["file_keys"] == "validation.root"
    assert derived["dataset"]["hdf5"]["file_keys"] == "validation.h5"
    assert derived["dataset"]["hdf5_key_map"] == {"old": "new"}
    assert "sampler" not in derived


@pytest.mark.parametrize(
    "loader,cfg,message",
    [
        (ordinary_loader(), {}, "file_keys"),
        (
            {
                **ordinary_loader(),
                "dataset": {
                    "name": "joint",
                    "primary": {},
                    "secondary": {},
                },
            },
            {"sources": {"primary": {"file_keys": "a.root"}}},
            "exactly",
        ),
        (
            {**ordinary_loader(), "dataset": "dataset.yaml"},
            {"file_keys": "a.root"},
            "inline dataset",
        ),
    ],
)
def test_build_loader_config_rejects_ambiguous_sources(loader, cfg, message):
    """Validation source errors should fail before any files are opened."""
    with pytest.raises((TypeError, ValueError), match=message):
        ValidationManager.build_loader_config(loader, cfg, seed=1)


def test_early_stopping_tracks_and_restores_progress():
    """Early stopping should apply delta/patience and serialize its progress."""
    stopping = EarlyStopping(monitor="loss", mode="min", patience=2, min_delta=0.1)

    assert not stopping.update({"loss": 2.0})
    assert not stopping.update({"loss": 1.95})
    assert stopping.update({"loss": 1.94})

    restored = EarlyStopping(
        monitor="loss",
        mode="min",
        patience=2,
        min_delta=0.1,
        state=stopping.state_dict(),
    )
    assert restored.best == 2.0
    assert restored.bad_checks == 2

    with pytest.raises(KeyError, match="accuracy"):
        restored.update({"accuracy": 1.0})


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"mode": "sideways"}, "mode"),
        ({"patience": -1}, "patience"),
        ({"min_delta": -0.1}, "min_delta"),
    ],
)
def test_early_stopping_validates_policy(kwargs, message):
    """Invalid early-stopping policies should fail during configuration."""
    with pytest.raises(ValueError, match=message):
        EarlyStopping(**kwargs)

    stopping = EarlyStopping(monitor="score", mode="max")
    assert not stopping.update({"score": 0.5})
    assert not stopping.update({"score": 0.6})
    assert stopping.best == 0.6

    state = stopping.state_dict()
    state["mode"] = "min"
    with pytest.raises(ValueError, match="does not match"):
        stopping.restore(state)


def test_best_checkpoint_tracks_and_restores_progress():
    """Best-checkpoint selection should promote only meaningful improvements."""
    policy = BestCheckpoint(
        monitor="accuracy",
        mode="max",
        min_delta=0.05,
        path="weights/best.ckpt",
    )

    assert policy.update({"accuracy": 0.5})
    assert not policy.update({"accuracy": 0.54})
    assert policy.update({"accuracy": 0.6})
    assert policy.path == "weights/best.ckpt"

    restored = BestCheckpoint(
        monitor="accuracy",
        mode="max",
        state=policy.state_dict(),
    )
    assert restored.best == pytest.approx(0.6)

    incompatible = policy.state_dict()
    incompatible["mode"] = "min"
    with pytest.raises(ValueError, match="does not match"):
        restored.restore(incompatible)
    with pytest.raises(KeyError, match="accuracy"):
        restored.update({"loss": 1.0})

    with pytest.raises(ValueError, match="mode"):
        BestCheckpoint(mode="sideways")
    with pytest.raises(ValueError, match="min_delta"):
        BestCheckpoint(min_delta=-1.0)
    with pytest.raises(TypeError, match="path"):
        BestCheckpoint(path=1)


def test_validation_manager_runs_fraction_logs_and_updates_stopping(
    monkeypatch, tmp_path
):
    """Validation should average scalar outputs over the deterministic fraction."""

    class FakeLoader:
        sampler = object()

        def __len__(self):
            return 4

    class FakeIO:
        def __init__(self, **_kwargs):
            self.loader = FakeLoader()
            self.values = iter([1.0, 3.0, 9.0, 12.0])
            self.closed = False
            self.watch = StopwatchManager()

        def __len__(self):
            return 4

        def reset_loader(self):
            self.values = iter([1.0, 3.0, 9.0, 12.0])

        def load(self):
            value = next(self.values)
            return {"index": np.array([int(value)]), "value": value}

        def close(self):
            self.closed = True

    class FakeModel:
        checkpoint_validation = None
        device = "cpu"

        @staticmethod
        def evaluate(data, iteration):
            return {"loss": data["value"] + iteration, "prediction": [1, 2]}

    monkeypatch.setattr(validation_mod, "IOManager", FakeIO)
    manager = ValidationManager(
        {
            "file_keys": "validation.root",
            "fraction": 0.5,
            "early_stopping": {"monitor": "loss", "patience": 1},
            "best_checkpoint": True,
        },
        ordinary_loader(),
        FakeModel(),
        rank=None,
        dtype="float32",
        world_size=0,
        distributed=False,
        seed=1,
        log_dir=str(tmp_path),
    )

    metrics = manager.run(iteration=2, epoch=0.75)
    assert metrics == {"loss": 4.0}
    log = tmp_path / "validation_log-0000003.csv"
    log_df = np.genfromtxt(log, delimiter=",", names=True)
    assert log_df.shape == (2,)
    assert log_df["iter"].tolist() == [0.0, 1.0]
    assert log_df["epoch"].tolist() == [0.75, 0.75]
    assert log_df["loss"].tolist() == [3.0, 5.0]
    assert manager.update_best_checkpoint(metrics)
    assert not manager.update_best_checkpoint(metrics)
    assert not manager.update_early_stopping(metrics)
    state = manager.checkpoint_state(metrics)
    assert state["metrics"] == metrics
    assert state["best_checkpoint"]["best"] == 4.0

    manager.log_dir = None
    manager.model.evaluate = lambda *_args: {"prediction": [1, 2]}
    with pytest.raises(RuntimeError, match="did not produce any scalar"):
        manager.run(iteration=3)
    manager.early_stopping = None
    manager.best_checkpoint = None
    assert not manager.update_early_stopping(metrics)
    assert not manager.update_best_checkpoint(metrics)
    manager.close()
    assert manager.io.closed


def test_validation_manager_validates_runtime_options(monkeypatch):
    """Fraction and early-stopping blocks should be checked before use."""

    class FakeLoader:
        def __len__(self):
            return 1

    class FakeIO:
        def __init__(self, **_kwargs):
            self.loader = FakeLoader()

    model = SimpleNamespace(checkpoint_validation=None)
    monkeypatch.setattr(validation_mod, "IOManager", FakeIO)
    kwargs = {
        "loader": ordinary_loader(),
        "model": model,
        "rank": None,
        "dtype": "float32",
        "world_size": 0,
        "distributed": False,
        "seed": 1,
    }

    with pytest.raises(TypeError, match="fraction"):
        ValidationManager({"file_keys": "val.root", "fraction": "half"}, **kwargs)
    with pytest.raises(ValueError, match="fraction"):
        ValidationManager({"file_keys": "val.root", "fraction": 0.0}, **kwargs)
    with pytest.raises(TypeError, match="early_stopping"):
        ValidationManager(
            {"file_keys": "val.root", "early_stopping": "invalid"}, **kwargs
        )
    with pytest.raises(TypeError, match="best_checkpoint"):
        ValidationManager(
            {"file_keys": "val.root", "best_checkpoint": "invalid"}, **kwargs
        )


def test_validation_log_uses_distributed_and_input_prefixes(monkeypatch, tmp_path):
    """Validation log names should mirror ordinary driver naming options."""
    calls = []
    manager = object.__new__(ValidationManager)
    manager.log_dir = None
    manager.distributed = True
    manager.rank = 2
    manager.prefix_log = True
    manager.overwrite_log = True
    manager.csv_buffer_size = 8
    manager.io = SimpleNamespace(
        format_log_name=lambda name, _directory: f"sample_{name}"
    )
    monkeypatch.setattr(
        validation_mod,
        "LogManager",
        lambda path, **kwargs: calls.append((path, kwargs)) or "logger",
    )

    assert manager.initialize_log(9) is None
    manager.log_dir = str(tmp_path / "logs")
    log_manager = manager.initialize_log(9)

    assert log_manager == "logger"
    assert calls == [
        (
            str(tmp_path / "logs" / "sample_validation_proc2_log-0000010.csv"),
            {"overwrite": True, "buffer_size": 8},
        )
    ]


def test_validation_source_configuration_errors():
    """Malformed simple and composite source overrides should be explicit."""
    with pytest.raises(ValueError, match="only valid for composite"):
        ValidationManager.build_loader_config(
            ordinary_loader(),
            {"sources": {"data": {"file_keys": "val.root"}}},
            seed=1,
        )

    joint = ordinary_loader()
    joint["dataset"] = {"name": "joint", "primary": {}, "secondary": {}}
    with pytest.raises(ValueError, match="named"):
        ValidationManager.build_loader_config(joint, {"file_keys": "val.root"}, seed=1)
    with pytest.raises(TypeError, match="primary"):
        ValidationManager.build_loader_config(
            joint,
            {
                "sources": {
                    "primary": "val.root",
                    "secondary": {"file_keys": "secondary.root"},
                }
            },
            seed=1,
        )
    with pytest.raises(KeyError, match="unknown"):
        ValidationManager.build_loader_config(
            ordinary_loader(),
            {"file_keys": "val.root", "unknown": True},
            seed=1,
        )
    with pytest.raises(KeyError, match="unexpected"):
        ValidationManager.validate_source(
            {"file_keys": "val.root", "unexpected": True}, "source"
        )

    joint["dataset"]["primary"] = "primary.yaml"
    with pytest.raises(TypeError, match="primary"):
        ValidationManager.build_loader_config(
            joint,
            {
                "sources": {
                    "primary": {"file_keys": "primary.root"},
                    "secondary": {"file_keys": "secondary.root"},
                }
            },
            seed=1,
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_validation_run_reduces_distributed_scalar_metrics(monkeypatch):
    """Distributed validation should seed its sampler and reduce each scalar."""
    epochs = []
    sampler = SimpleNamespace(set_epoch=lambda epoch: epochs.append(epoch))

    class FakeIO:
        loader = SimpleNamespace(sampler=sampler)

        @staticmethod
        def reset_loader():
            pass

        @staticmethod
        def load():
            return {}

    manager = object.__new__(ValidationManager)
    manager.io = FakeIO()
    manager.model = SimpleNamespace(
        device="cpu",
        evaluate=lambda data, iteration: {
            "loss": validation_mod.torch.tensor(2.0),
            "score": np.asarray([4.0]),
        },
    )
    manager.distributed = True
    manager.num_iterations = 1
    monkeypatch.setattr(
        validation_mod.torch.distributed,
        "all_reduce",
        lambda value: value.mul_(2.0),
    )

    assert manager.run(3) == {"loss": 2.0, "score": 4.0}
    assert epochs == [0]

    manager.distributed = False
    manager.model.evaluate = lambda data, iteration: {"prediction": [1, 2]}
    with pytest.raises(RuntimeError, match="scalar metrics"):
        manager.run(3)

    manager.early_stopping = None
    manager.best_checkpoint = None
    assert not manager.update_early_stopping({"loss": 1.0})
    assert not manager.update_best_checkpoint({"loss": 1.0})
    assert manager.checkpoint_state({"loss": 1.0}) == {"metrics": {"loss": 1.0}}
