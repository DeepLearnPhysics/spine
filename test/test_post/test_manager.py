from __future__ import annotations

from typing import cast

import pytest

import spine.post.manager as manager_mod
from spine.post.manager import PostManager


class FakePostModule:
    def __init__(self, name, cfg, calls):
        self.name = name
        self.cfg = cfg
        self.calls = calls
        self._upstream = tuple(cfg.get("upstream", ()))

    def __call__(self, data, entry=None):
        self.calls.append((self.name, entry))
        if entry is None:
            return {"value": data["index"] + self.cfg.get("offset", 0)}
        return {"value": data["index"][entry] + self.cfg.get("offset", 0)}


def test_post_manager_parses_priority_names_and_keeps_config_clean(monkeypatch):
    calls = []
    monkeypatch.setattr(
        manager_mod,
        "post_processor_factory",
        lambda name, cfg, parent_path=None: FakePostModule(
            name, {**cfg, "parent_path": parent_path}, calls
        ),
    )
    cfg = {
        "low": {"name": "processor", "priority": 1, "offset": 1},
        "high": {"name": "processor", "priority": 3, "offset": 2},
    }

    manager = PostManager(cfg, parent_path="config")

    assert list(manager.modules) == ["high", "low"]
    assert cast(FakePostModule, manager.modules["high"]).cfg == {
        "offset": 2,
        "parent_path": "config",
    }
    assert cfg["high"]["priority"] == 3
    assert cfg["high"]["name"] == "processor"


def test_post_manager_runs_batches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        manager_mod,
        "post_processor_factory",
        lambda name, cfg, parent_path=None: FakePostModule(name, cfg, calls),
    )
    manager = PostManager({"demo": {"offset": 10}})
    data = {"index": [1, 2, 3]}

    manager(data)

    assert data["value"] == [11, 12, 13]
    assert calls == [("demo", 0), ("demo", 1), ("demo", 2)]


def test_post_manager_checks_dependencies_against_labels_and_names(monkeypatch):
    calls = []
    monkeypatch.setattr(
        manager_mod,
        "post_processor_factory",
        lambda name, cfg, parent_path=None: FakePostModule(name, cfg, calls),
    )

    PostManager(
        {
            "custom_source": {"name": "source"},
            "custom_child": {"name": "child", "upstream": ("source",)},
        },
        post_list=[],
    )
    PostManager(
        {
            "custom_source": {"name": "source"},
            "custom_child": {"name": "child", "upstream": ("custom_source",)},
        },
        post_list=[],
    )

    with pytest.raises(ValueError, match="missing an essential upstream"):
        PostManager(
            {"custom_child": {"name": "child", "upstream": ("source",)}},
            post_list=[],
        )
