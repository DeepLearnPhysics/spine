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


def test_ordered_post_matches_priority_order_and_executes_in_sequence(monkeypatch):
    """Repeated provider instances keep their own names, configs, and timers."""
    from spine.config import load_config

    calls = []

    class Processor:
        _upstream = ()

        def __init__(self, provider, config, parent_path):
            self.provider = provider
            self.config = config
            self.parent_path = parent_path

        def __call__(self, data, entry=None):
            calls.append(self.config)
            return {
                "value": data["value"] * self.config["factor"] + self.config["offset"]
            }

    monkeypatch.setattr(manager_mod, "post_processor_factory", Processor)
    legacy = PostManager(
        {
            "add": {"name": "arithmetic", "priority": 1, "factor": 1, "offset": 1},
            "scale": {"name": "arithmetic", "priority": 2, "factor": 2, "offset": 0},
        },
        parent_path="config",
    )
    config = load_config("""
post:
  stages:
    - name: add
      provider: arithmetic
      config: {factor: 1, offset: 1}
override:
  post.stages~:
    insert:
      before: add
      value:
        name: scale
        provider: arithmetic
        config: {factor: 2, offset: 0}
""")
    ordered = PostManager(config["post"], parent_path="config")
    assert list(ordered.modules) == list(legacy.modules) == ["scale", "add"]
    assert ordered.module_names == legacy.module_names == ("arithmetic", "arithmetic")
    for manager in (legacy, ordered):
        data = {"index": 0, "value": 10}
        manager(data)
        assert data["value"] == 21
        assert all(
            module.parent_path == "config" for module in manager.modules.values()
        )
    assert (
        calls[:2]
        == calls[2:]
        == [{"factor": 2, "offset": 0}, {"factor": 1, "offset": 1}]
    )
    assert "priority" not in ordered.modules["scale"].config


@pytest.mark.parametrize("dependency", ["source", "custom_source"])
def test_ordered_post_dependency_checks(monkeypatch, dependency):
    """Validate earlier providers/instances while retaining the None opt-out."""
    monkeypatch.setattr(
        manager_mod,
        "post_processor_factory",
        lambda name, cfg, parent_path=None: FakePostModule(name, cfg, []),
    )
    source = {"name": "custom_source", "provider": "source"}
    child = {
        "name": "custom_child",
        "provider": "child",
        "config": {"upstream": [dependency]},
    }
    PostManager({"stages": [source, child]}, post_list=[])
    PostManager({"stages": [child]}, post_list=[dependency])
    PostManager({"stages": [child]}, post_list=None)
    with pytest.raises(ValueError, match="missing an essential upstream"):
        PostManager({"stages": [child, source]}, post_list=[])
    with pytest.raises(ValueError, match="missing an essential upstream"):
        PostManager({"stages": [child]}, post_list=[])
    with pytest.raises(ValueError, match="Cannot mix"):
        PostManager({"stages": [source], "legacy": {}})
    with pytest.raises(ValueError, match="priority"):
        PostManager({"stages": [{**source, "priority": 1}]})


def test_ordered_post_constructs_registered_provider():
    """Instance labels are not passed to the real provider factory as names."""
    from spine.post.reco.direction import DirectionProcessor

    manager = PostManager(
        {
            "stages": [
                {
                    "name": "particle_directions",
                    "provider": "direction",
                    "config": {"radius": 5.0},
                }
            ]
        },
        post_list=[],
    )
    assert isinstance(manager.modules["particle_directions"], DirectionProcessor)
    assert manager.modules["particle_directions"].radius == 5.0
    assert manager.module_names == ("direction",)


@pytest.mark.parametrize("ordered", [False, True])
def test_post_stage_cannot_satisfy_its_own_upstream_dependency(monkeypatch, ordered):
    """Only previously configured or externally recorded processors count."""
    monkeypatch.setattr(
        manager_mod,
        "post_processor_factory",
        lambda name, cfg, parent_path=None: FakePostModule(name, cfg, []),
    )
    cfg = {"self": {"name": "child", "upstream": ["self"]}}
    if ordered:
        cfg = {
            "stages": [
                {"name": "self", "provider": "child", "config": {"upstream": ["self"]}}
            ]
        }
    with pytest.raises(ValueError, match="missing an essential upstream"):
        PostManager(cfg, post_list=[])
    PostManager(cfg, post_list=["self"])
