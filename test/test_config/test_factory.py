from collections import OrderedDict

import pytest

from spine.config.factory import (
    instantiate,
    instantiate_modules,
    module_dict,
    parse_module_config,
)


class Alpha:
    name = "alpha"

    def __init__(self, value=0):
        self.value = value


class Beta:
    name = "beta"

    def __init__(self, value=0):
        self.value = value


class Broken:
    """Constructor that exposes factory error logging and re-raising."""

    name = "broken"

    def __init__(self):
        raise RuntimeError("construction failed")


def test_parse_module_config_uses_key_as_default_name_and_preserves_order():
    parsed = parse_module_config(
        OrderedDict(
            [
                ("alpha", {"value": 1}),
                ("second", {"name": "beta", "value": 2}),
            ]
        )
    )

    assert list(parsed) == ["alpha", "second"]
    assert parsed["alpha"] == {"name": "alpha", "cfg": {"value": 1}, "priority": None}
    assert parsed["second"] == {"name": "beta", "cfg": {"value": 2}, "priority": None}


def test_parse_module_config_can_sort_lower_priority_first():
    parsed = parse_module_config(
        OrderedDict(
            [
                ("late", {"name": "alpha", "priority": 20}),
                ("early", {"name": "beta", "priority": 10}),
                ("default_order", {"name": "alpha"}),
            ]
        ),
        sort_by_priority=True,
    )

    assert list(parsed) == ["early", "late", "default_order"]


def test_parse_module_config_can_sort_higher_priority_first():
    parsed = parse_module_config(
        OrderedDict(
            [
                ("late", {"name": "alpha", "priority": 10}),
                ("early", {"name": "beta", "priority": 20}),
                ("default_order", {"name": "alpha"}),
            ]
        ),
        sort_by_priority=True,
        priority_descending=True,
    )

    assert list(parsed) == ["early", "late", "default_order"]


def test_parse_module_config_validates_blocks():
    with pytest.raises(TypeError, match="must be a mapping"):
        parse_module_config([])

    with pytest.raises(TypeError, match="Configuration for module"):
        parse_module_config({"alpha": "bad"})


def test_instantiate_modules_returns_label_to_instance_mapping():
    modules = instantiate_modules(
        {"alpha": Alpha, "beta": Beta},
        {
            "first": {"name": "alpha", "value": 1},
            "beta": {"value": 2},
        },
    )

    assert list(modules) == ["first", "beta"]
    assert isinstance(modules["first"], Alpha)
    assert modules["first"].value == 1
    assert isinstance(modules["beta"], Beta)
    assert modules["beta"].value == 2


def test_instantiate_validates_name_keys_and_duplicate_kwargs():
    registry = {"alpha": Alpha}

    assert isinstance(instantiate(registry, "alpha"), Alpha)

    with pytest.raises(ValueError, match="one of"):
        instantiate(registry, {"name": "alpha", "parser": "alpha"}, alt_name="parser")

    instance = instantiate(
        registry,
        {"parser": "alpha", "value": 4},
        alt_name="parser",
    )
    assert instance.value == 4

    with pytest.raises(ValueError, match="under `name`"):
        instantiate(registry, {"value": 1})

    with pytest.raises(ValueError, match="Available names.*alpha"):
        instantiate(registry, "missing")

    with pytest.warns(DeprecationWarning, match="keyword arguments"):
        with pytest.raises(ValueError, match="under `args` and `kwargs`"):
            instantiate(registry, {"name": "alpha", "args": {"value": 1}}, value=2)

    with pytest.raises(ValueError, match="top level and under `kwargs`"):
        instantiate(registry, {"name": "alpha", "value": 1}, value=2)

    with pytest.deprecated_call(match="keyword arguments"):
        instance = instantiate(registry, {"name": "alpha", "args": {"value": 3}})
    assert instance.value == 3


def test_module_dictionary_alias_warning_and_factory_error_paths():
    """Aliases and failed constructors should retain diagnostic behavior."""
    import sys

    module = sys.modules[__name__]
    Alpha.aliases = ("old_alpha",)
    try:
        with pytest.deprecated_call(match="deprecated"):
            registry = module_dict(module, class_name="old_alpha")
        assert registry["old_alpha"] is Alpha

        registry = module_dict(module)
        assert registry["old_alpha"] is Alpha
        assert module_dict(module, pattern="Alpha") == {
            "Alpha": Alpha,
            "alpha": Alpha,
            "old_alpha": Alpha,
        }
    finally:
        del Alpha.aliases

    with pytest.raises(RuntimeError, match="construction failed"):
        instantiate({"broken": Broken}, {"name": "broken"})


def test_parse_module_config_skips_none_by_default():
    """Disabled optional modules should be skipped without disturbing order."""
    assert parse_module_config({"disabled": None, "alpha": {}}) == {
        "alpha": {"name": "alpha", "cfg": {}, "priority": None}
    }
