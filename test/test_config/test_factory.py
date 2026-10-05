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


def test_ordered_module_stages_preserve_identity_order_and_inputs():
    """Explicit stages normalize like mappings without priority sorting."""
    stages = [
        {"name": "second", "provider": "alpha", "config": {"nested": [2]}},
        {"name": "first", "provider": "alpha"},
    ]
    parsed = parse_module_config(
        {"stages": stages},
        stages_key="stages",
        sort_by_priority=True,
        priority_descending=True,
    )
    assert list(parsed) == ["second", "first"]
    assert parsed["second"] == {
        "name": "alpha",
        "cfg": {"nested": [2]},
        "priority": None,
    }
    assert parsed["first"] == {"name": "alpha", "cfg": {}, "priority": None}
    parsed["second"]["cfg"]["nested"].append(3)
    assert stages[0]["config"] == {"nested": [2]}
    assert parse_module_config({"stages": []}, stages_key="stages") == {}
    # Opt-in avoids reinterpreting a legacy module literally named 'stages'.
    assert (
        parse_module_config({"stages": {"name": "alpha"}})["stages"]["name"] == "alpha"
    )


@pytest.mark.parametrize(
    ("stages", "error", "match"),
    [
        (None, TypeError, "must be a list"),
        ({}, TypeError, "must be a list"),
        ([None], TypeError, "must be a mapping"),
        ([{}], ValueError, "nonempty `name`"),
        ([{"provider": "alpha"}], ValueError, "nonempty `name`"),
        ([{"name": "a", "provider": None}], ValueError, "nonempty `provider`"),
        ([{"name": "a", "provider": ""}], ValueError, "nonempty `provider`"),
        ([{"name": "a", "provider": " "}], ValueError, "nonempty `provider`"),
        ([{"name": " ", "provider": "alpha"}], ValueError, "nonempty `name`"),
        ([{"name": "a", "provider": 1}], ValueError, "nonempty `provider`"),
        (
            [{"name": "a", "provider": "alpha", "priority": None}],
            ValueError,
            "priority",
        ),
        (
            [{"name": "a", "provider": "alpha", "config": {}, "value": 1}],
            ValueError,
            "cannot mix",
        ),
        (
            [{"name": "a", "provider": "alpha", "config": None}],
            TypeError,
            "must be a mapping",
        ),
        ([{"name": "a", "provider": "alpha"}] * 2, ValueError, "Duplicate"),
    ],
)
def test_ordered_module_stages_validate_structure(stages, error, match):
    """Both managers share strict ordered-entry validation."""
    with pytest.raises(error, match=match):
        parse_module_config({"stages": stages}, stages_key="stages")


def test_ordered_module_stages_reject_mixed_formats():
    """Even disabled legacy entries cannot accompany an explicit stage list."""
    with pytest.raises(ValueError, match="Cannot mix"):
        parse_module_config({"stages": [], "alpha": None}, stages_key="stages")


def test_instantiate_modules_accepts_shared_ordered_schema():
    """Generic consumers can instantiate lists without manager-specific parsing."""
    instances = instantiate_modules(
        {"alpha": Alpha},
        {
            "stages": [
                {"name": "second", "provider": "alpha", "config": {"value": 2}},
                {"name": "first", "provider": "alpha", "config": {"value": 1}},
            ]
        },
        stages_key="stages",
        sort_by_priority=True,
    )
    assert list(instances) == ["second", "first"]
    assert [instance.value for instance in instances.values()] == [2, 1]


def test_priority_parsing_can_be_disabled_for_legacy_consumers():
    """Managers without scheduling metadata leave provider parameters intact."""
    parsed = parse_module_config({"alpha": {"priority": 7}}, priority_key=None)
    assert parsed["alpha"] == {
        "name": "alpha",
        "cfg": {"priority": 7},
        "priority": None,
    }


def test_ordered_module_provider_defaults_to_required_instance_name():
    """Shorthand and explicit providers share identity and duplicate rules."""
    stages = [
        {"name": "alpha", "config": {"value": 1}},
        {"name": "another_alpha", "provider": "alpha", "config": {"value": 2}},
    ]
    instances = instantiate_modules(
        {"alpha": Alpha}, {"stages": stages}, stages_key="stages"
    )
    assert list(instances) == ["alpha", "another_alpha"]
    assert [instance.value for instance in instances.values()] == [1, 2]
    assert "provider" not in stages[0]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_module_config(
            {"stages": [{"name": "alpha"}, {"name": "alpha", "provider": "beta"}]},
            stages_key="stages",
        )


@pytest.mark.parametrize("inline", [False, True])
def test_stage_parameters_accept_inline_or_nested_forms(inline):
    """Both representations produce identical provider constructor arguments."""
    parameters = {"value": 4}
    stage = {"name": "alpha", **(parameters if inline else {"config": parameters})}
    instances = instantiate_modules(
        {"alpha": Alpha}, {"stages": [stage]}, stages_key="stages"
    )
    assert instances["alpha"].value == 4
    assert parameters == {"value": 4}


@pytest.mark.parametrize("nested", [{}, {"value": 1}, {"other": 2}])
def test_stage_parameters_reject_mixing_even_without_overlap(nested):
    """No precedence is inferred for new ordered module consumers."""
    with pytest.raises(ValueError, match="cannot mix"):
        parse_module_config(
            {"stages": [{"name": "alpha", "value": 4, "config": nested}]},
            stages_key="stages",
        )


def test_parameter_extraction_copies_inputs_and_preserves_nested_reserved_fields():
    """Extraction does not mutate payloads or consume nested provider options."""
    from spine.config.factory import extract_module_parameters

    for descriptor in ({"values": [1]}, {"config": {"values": [1]}}):
        parsed = extract_module_parameters(descriptor, context="Test stage")
        parsed["values"].append(2)
        original = descriptor.get("config", descriptor)
        assert original["values"] == [1]
    reserved = {
        "name": "parameter_name",
        "provider": "parameter_provider",
        "priority": 3,
        "config": {"option": True},
    }
    parsed = parse_module_config(
        {"stages": [{"name": "alpha", "config": reserved}]}, stages_key="stages"
    )
    assert parsed["alpha"]["cfg"] == reserved


def test_inline_unknown_parameters_are_validated_by_provider():
    """Unknown inline keys reach constructor validation instead of being dropped."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        instantiate_modules(
            {"alpha": Alpha},
            {"stages": [{"name": "alpha", "typo": 1}]},
            stages_key="stages",
        )
