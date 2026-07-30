"""Unit tests for model discovery and manager configuration handling."""

import pytest

from spine.model.manager import ModelManager
from spine.utils.conditional import TORCH_AVAILABLE, torch


class FakeWatch:
    """Minimal stopwatch payload used by the manager reset test."""

    def __init__(self):
        self.running = False
        self.paused = False


class FakeWatchManager:
    """Minimal stopwatch manager with call recording."""

    def __init__(self):
        self.calls = []
        self.watches = {}

    def initialize(self, key):
        self.calls.append(("initialize", key))
        self.watches.setdefault(key, FakeWatch())

    def start(self, key):
        self.calls.append(("start", key))
        self.watches.setdefault(key, FakeWatch()).running = True

    def stop(self, key):
        self.calls.append(("stop", key))
        self.watches.setdefault(key, FakeWatch()).running = False

    def reset(self):
        self.calls.append(("reset", None))
        for watch in self.watches.values():
            watch.running = False
            watch.paused = False

    def reset_if_active(self):
        for watch in self.watches.values():
            if watch.running or watch.paused:
                self.reset()
                break


def test_model_manager_resets_stale_watch_before_call():
    """ModelManager clears stale stopwatch state before forwarding."""
    manager = object.__new__(ModelManager)
    manager.train = False
    manager.to_numpy = False
    manager.watch = FakeWatchManager()
    manager.watch.initialize("forward")
    manager.watch.start("forward")
    manager.forward = lambda data, iteration: {"value": data["index"] + iteration}

    result = manager({"index": 2}, iteration=3)

    assert result == {"value": 5}
    assert manager.watch.calls[:4] == [
        ("initialize", "forward"),
        ("start", "forward"),
        ("reset", None),
        ("start", "forward"),
    ]
    assert manager.watch.calls[-1] == ("stop", "forward")


def test_clean_config_returns_sanitized_copy():
    """Manager-only weight settings must not leak into model constructors."""

    modules = {
        "backbone": {
            "depth": 5,
            "weight_path": "weights.ckpt",
            "freeze_weights": True,
            "nested": [{"model_name": "old_name", "width": 32}],
        }
    }

    cleaned = ModelManager.clean_config(modules)

    assert cleaned == {"backbone": {"depth": 5, "nested": [{"width": 32}]}}
    assert modules["backbone"]["weight_path"] == "weights.ckpt"
    assert modules["backbone"]["nested"][0]["model_name"] == "old_name"


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_inference_manager_does_not_construct_loss(monkeypatch):
    """Pure inference should not require loss configuration or construction."""

    loss_calls = []

    class TestNetwork(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.linear = torch.nn.Linear(network["width"], 1)

        def forward(self, data):
            return {"prediction": self.linear(data)}

    class TestLoss(torch.nn.Module):
        def __init__(self, **modules):
            super().__init__()
            loss_calls.append(modules)

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda name: (TestNetwork, TestLoss),
    )

    modules = {
        "network": {"width": 2},
        "network_loss": {"reduction": "mean"},
    }
    manager = ModelManager(
        name="test",
        modules=modules,
        network_input={"data": "data"},
    )

    assert manager.loss_fn is None
    assert loss_calls == []
    assert modules == {
        "network": {"width": 2},
        "network_loss": {"reduction": "mean"},
    }


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_manager_partitions_network_and_loss_configuration(monkeypatch):
    """Networks receive model blocks while losses receive the complete config."""

    network_calls = []
    loss_calls = []

    class TestNetwork(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            network_calls.append(network)

    class TestLoss(torch.nn.Module):
        def __init__(self, network, network_loss):
            super().__init__()
            loss_calls.append((network, network_loss))

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda name: (TestNetwork, TestLoss),
    )

    modules = {
        "network": {"width": 2},
        "network_loss": {"reduction": "mean"},
    }
    ModelManager(
        name="test",
        modules=modules,
        network_input={"data": "data"},
        loss_input={"target": "target"},
    )

    assert network_calls == [{"width": 2}]
    assert loss_calls == [({"width": 2}, {"reduction": "mean"})]
