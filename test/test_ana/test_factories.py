from __future__ import annotations

import spine.ana.factories as factories


class DummyAna:
    name = "dummy"

    def __init__(
        self,
        value=0,
        overwrite=False,
        log_dir=None,
        prefix=None,
        buffer_size=1,
    ):
        self.value = value
        self.overwrite = overwrite
        self.log_dir = log_dir
        self.prefix = prefix
        self.buffer_size = buffer_size


def test_ana_script_factory_does_not_mutate_config(monkeypatch):
    monkeypatch.setattr(factories, "ANA_DICT", {"dummy": DummyAna})
    cfg = {"value": 3}

    module = factories.ana_script_factory(
        "dummy",
        cfg,
        overwrite=True,
        log_dir="logs",
        prefix="input",
        buffer_size=8,
    )

    assert cfg == {"value": 3}
    assert module.value == 3
    assert module.overwrite is True
    assert module.log_dir == "logs"
    assert module.prefix == "input"
    assert module.buffer_size == 8


def test_ana_script_factory_omits_overwrite_when_unspecified(monkeypatch):
    monkeypatch.setattr(factories, "ANA_DICT", {"dummy": DummyAna})

    module = factories.ana_script_factory("dummy", {"value": 4})

    assert module.value == 4
    assert module.overwrite is False
