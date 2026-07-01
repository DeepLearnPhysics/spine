from collections import OrderedDict

import pytest

from spine.utils.factory import instantiate, instantiate_modules, parse_module_config


class Alpha:
    name = "alpha"

    def __init__(self, value=0):
        self.value = value


class Beta:
    name = "beta"

    def __init__(self, value=0):
        self.value = value


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

    with pytest.raises(ValueError, match="one of"):
        instantiate(registry, {"name": "alpha", "parser": "alpha"}, alt_name="parser")

    with pytest.raises(ValueError, match="under `name`"):
        instantiate(registry, {"value": 1})

    with pytest.raises(ValueError, match="under `args` and `kwargs`"):
        instantiate(registry, {"name": "alpha", "args": {"value": 1}}, value=2)

    with pytest.raises(ValueError, match="top level and under `kwargs`"):
        instantiate(registry, {"name": "alpha", "value": 1}, value=2)
