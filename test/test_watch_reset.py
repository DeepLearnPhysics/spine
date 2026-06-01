"""Tests for manager-local stopwatch reset behavior."""

from __future__ import annotations

from collections import OrderedDict

from spine.ana.manager import AnaManager
from spine.model.manager import ModelManager
from spine.post.manager import PostManager


class FakeWatch:
    """Minimal stopwatch payload used by manager tests."""

    def __init__(self) -> None:
        self.running = False
        self.paused = False


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


class FakeAnaModule:
    """Analysis-module stub with the writer lifecycle AnaManager expects."""

    def __call__(self, data, entry=None):
        return {"ana_value": data["index"] + 2}

    def close_writers(self):
        """No-op close hook for AnaManager cleanup."""


def test_model_manager_resets_stale_watch_before_call():
    """ModelManager should clear stale watch state before forwarding."""
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


def test_post_manager_resets_stale_watch_before_call():
    """PostManager should clear stale watch state before executing modules."""
    manager = object.__new__(PostManager)
    manager.watch = FakeWatchManager()
    manager.watch.initialize("demo")
    manager.watch.start("demo")
    manager.modules = OrderedDict(
        [("demo", lambda data, entry=None: {"post_value": data["index"] + 1})]
    )

    data = {"index": 7}
    manager(data)

    assert data["post_value"] == 8
    assert manager.watch.calls[:4] == [
        ("initialize", "demo"),
        ("start", "demo"),
        ("reset", None),
        ("start", "demo"),
    ]
    assert manager.watch.calls[-1] == ("stop", "demo")


def test_ana_manager_resets_stale_watch_before_call():
    """AnaManager should clear stale watch state before executing modules."""
    manager = object.__new__(AnaManager)
    manager.watch = FakeWatchManager()
    manager.watch.initialize("demo")
    manager.watch.start("demo")
    manager.modules = OrderedDict([("demo", FakeAnaModule())])

    data = {"index": 4}
    manager(data)

    assert data["ana_value"] == 6
    assert manager.watch.calls[:4] == [
        ("initialize", "demo"),
        ("start", "demo"),
        ("reset", None),
        ("start", "demo"),
    ]
    assert manager.watch.calls[-1] == ("stop", "demo")
