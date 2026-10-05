from __future__ import annotations

import pytest

import spine.ana.manager as manager_mod
from spine.ana.manager import AnaManager


class FakeAnaModule:
    def __init__(self, name, cfg, calls):
        self.name = name
        self.cfg = cfg
        self.calls = calls
        self.closed = False
        self.flushed = False

    def __call__(self, data, entry=None):
        self.calls.append((self.name, entry))
        if entry is None:
            return {"value": data["index"] + self.cfg["offset"]}
        return {"value": data["index"][entry] + self.cfg["offset"]}

    @property
    def supports_columnar(self):
        return self.cfg.get("columnar", False)

    def run_columnar(self, data):
        self.calls.append((self.name, "columnar"))
        return {"value": [value + self.cfg["offset"] for value in data["index"]]}

    def columnar_requests(self):
        return self.cfg.get("requests", {})

    def close_writers(self):
        self.closed = True

    def flush_writers(self):
        self.flushed = True


def test_ana_manager_parses_priority_names_and_keeps_config_clean(monkeypatch):
    calls = []

    def factory(name, cfg, overwrite, log_dir, prefix, buffer_size):
        return FakeAnaModule(
            name,
            {
                **cfg,
                "overwrite": overwrite,
                "log_dir": log_dir,
                "prefix": prefix,
                "buffer_size": buffer_size,
            },
            calls,
        )

    monkeypatch.setattr(manager_mod, "ana_script_factory", factory)
    cfg = {
        "low": {"name": "script", "priority": 1, "offset": 1},
        "high": {"name": "script", "priority": 3, "offset": 2},
        "overwrite": True,
        "prefix_output": True,
        "buffer_size": 4,
    }

    manager = AnaManager(cfg, log_dir="logs", prefix="input")

    assert list(manager.modules) == ["high", "low"]
    assert manager.modules["high"].cfg == {
        "offset": 2,
        "overwrite": True,
        "log_dir": "logs",
        "prefix": "input",
        "buffer_size": 4,
    }
    assert cfg["high"]["priority"] == 3
    assert cfg["high"]["name"] == "script"


def test_ana_manager_runs_batches_and_writer_lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(
        manager_mod,
        "ana_script_factory",
        lambda name, cfg, *args: FakeAnaModule(name, cfg, calls),
    )
    manager = AnaManager({"demo": {"offset": 10}})
    data = {"index": [1, 2, 3]}

    manager(data)
    manager.flush()
    manager.close()

    assert data["value"] == [11, 12, 13]
    assert calls == [("demo", 0), ("demo", 1), ("demo", 2)]
    assert manager.modules["demo"].flushed
    assert manager.modules["demo"].closed


def test_ana_manager_validates_global_options():
    with pytest.raises(TypeError, match="overwrite"):
        AnaManager({"overwrite": "yes"})

    with pytest.raises(TypeError, match="prefix_output"):
        AnaManager({"prefix_output": 1})

    with pytest.raises(TypeError, match="buffer_size"):
        AnaManager({"buffer_size": 1.5})

    with pytest.raises(TypeError, match="columnar"):
        AnaManager({}, columnar=1)


def test_ana_manager_uses_buffered_analysis_output_by_default(monkeypatch):
    """Analysis writers should use system buffering unless overridden."""
    buffer_sizes = []

    def factory(name, cfg, overwrite, log_dir, prefix, buffer_size):
        buffer_sizes.append(buffer_size)
        return FakeAnaModule(name, cfg, [])

    monkeypatch.setattr(manager_mod, "ana_script_factory", factory)
    AnaManager({"demo": {"offset": 0}})

    assert buffer_sizes == [-1]


def test_ana_manager_runs_supported_columnar_modules(monkeypatch):
    calls = []
    monkeypatch.setattr(
        manager_mod,
        "ana_script_factory",
        lambda name, cfg, *args: FakeAnaModule(name, cfg, calls),
    )
    manager = AnaManager(
        {"demo": {"offset": 10, "columnar": True}},
        columnar=True,
    )
    data = {"index": [1, 2, 3]}

    assert manager.supports_columnar
    manager.process_columnar(data)

    assert data["value"] == [11, 12, 13]
    assert calls == [("demo", "columnar")]


def test_ana_manager_preflights_unsupported_columnar_modules(monkeypatch):
    calls = []
    monkeypatch.setattr(
        manager_mod,
        "ana_script_factory",
        lambda name, cfg, *args: FakeAnaModule(name, cfg, calls),
    )
    with pytest.raises(ValueError, match="event_only"):
        AnaManager(
            {
                "supported": {"priority": 2, "offset": 10, "columnar": True},
                "event_only": {"priority": 1, "offset": 20},
            },
            columnar=True,
        )
    assert calls == []


def test_ana_manager_rejects_columnar_call_in_event_mode(monkeypatch):
    """An event-mode manager should not accept columnar execution."""
    calls = []
    monkeypatch.setattr(
        manager_mod,
        "ana_script_factory",
        lambda name, cfg, *args: FakeAnaModule(name, cfg, calls),
    )
    manager = AnaManager({"demo": {"offset": 10, "columnar": True}})

    with pytest.raises(RuntimeError, match="not configured"):
        manager.process_columnar({"index": [1, 2, 3]})


def test_ana_manager_merges_columnar_requests(monkeypatch):
    """Repeated projections should union fields and preserve requiredness."""
    calls = []
    monkeypatch.setattr(
        manager_mod,
        "ana_script_factory",
        lambda name, cfg, *args: FakeAnaModule(name, cfg, calls),
    )
    manager = AnaManager(
        {
            "first": {
                "requests": {
                    "particles": (("id",), False),
                    "all_fields": (None, False),
                }
            },
            "second": {
                "requests": {
                    "particles": (("pid",), True),
                    "all_fields": (("size",), True),
                }
            },
        }
    )

    assert manager.columnar_requests() == {
        "particles": (("id", "pid"), True),
        "all_fields": (None, True),
    }


@pytest.mark.parametrize("columnar", [False, True])
def test_ordered_analysis_settings_execution_and_lifecycle(monkeypatch, columnar):
    """The shared schema preserves analysis settings and both execution modes."""
    from spine.config import load_config

    calls = []
    settings = []

    def factory(name, cfg, overwrite, log_dir, prefix, buffer_size):
        settings.append((overwrite, log_dir, prefix, buffer_size))
        return FakeAnaModule(name, cfg, calls)

    monkeypatch.setattr(manager_mod, "ana_script_factory", factory)
    cfg = load_config("""
ana:
  overwrite: true
  prefix_output: true
  buffer_size: 8
  stages:
    - name: last
      provider: script
      config: {offset: 20, columnar: true}
override:
  ana.stages~:
    insert:
      before: last
      value:
        name: first
        provider: script
        config: {offset: 10, columnar: true}
""")["ana"]
    manager = AnaManager(cfg, log_dir="logs", prefix="input", columnar=columnar)
    assert list(manager.modules) == ["first", "last"]
    assert settings == [(True, "logs", "input", 8)] * 2
    data = {"index": [1, 2]}
    if columnar:
        manager.process_columnar(data)
    else:
        manager(data)
    assert data["value"] == [21, 22]
    manager.flush()
    manager.close()
    assert all(module.flushed and module.closed for module in manager.modules.values())
    assert cfg["stages"][0]["config"] == {"offset": 10, "columnar": True}


def test_ordered_analysis_validation(monkeypatch):
    """Ordered analysis uses common validation and columnar preflight."""
    monkeypatch.setattr(
        manager_mod,
        "ana_script_factory",
        lambda name, cfg, *args: FakeAnaModule(name, cfg, []),
    )
    stage = {"name": "event_only", "provider": "script", "config": {"offset": 1}}
    with pytest.raises(ValueError, match="priority"):
        AnaManager({"stages": [{**stage, "priority": 1}]})
    with pytest.raises(ValueError, match="Cannot mix"):
        AnaManager({"stages": [stage], "legacy": {}})
    with pytest.raises(ValueError, match="event_only"):
        AnaManager({"stages": [stage]}, columnar=True)
