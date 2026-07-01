from __future__ import annotations

import spine.post.factories as factories


class DummyPost:
    name = "dummy"
    provide_parent_path = True

    def __init__(self, value=0, parent_path=None):
        self.value = value
        self.parent_path = parent_path


def test_post_processor_factory_does_not_mutate_config(monkeypatch):
    monkeypatch.setattr(factories, "POST_DICT", {"dummy": DummyPost})
    cfg = {"value": 3}

    module = factories.post_processor_factory("dummy", cfg, parent_path="config")

    assert cfg == {"value": 3}
    assert module.value == 3
    assert module.parent_path == "config"
