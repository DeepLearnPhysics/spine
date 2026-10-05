"""Missing targets retain order, source policy, and explicit optionality."""

import warnings

import pytest
import yaml

from spine.config import load_config, load_config_file
from spine.config.errors import ConfigPathError, ConfigTypeError, ConfigValidationError


def write_config(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return str(path)


def test_deferred_assignment_cannot_overwrite_later_sibling(tmp_path):
    early = write_config(
        tmp_path, "early.yaml", {"override": {"target.value": "early"}}
    )
    late = write_config(
        tmp_path,
        "late.yaml",
        {
            "target": {"value": "initial"},
            "override": {"target.value": "late"},
        },
    )
    result = load_config(yaml.safe_dump({"include": [early, late]}))
    assert result["target"]["value"] == "late"


def test_repeated_deferred_appends_and_source_append_modes(tmp_path):
    first = write_config(
        tmp_path,
        "first.yaml",
        {
            "__meta__": {"kind": "fragment", "list_append": "append"},
            "override": {"target.items+": ["a", "a"]},
        },
    )
    second = write_config(
        tmp_path,
        "second.yaml",
        {
            "__meta__": {"kind": "fragment", "list_append": "unique"},
            "override": {"target.items+": ["a", "b"]},
        },
    )
    middle = write_config(tmp_path, "middle.yaml", {"include": [first, second]})
    result = load_config(
        yaml.safe_dump(
            {
                "include": middle,
                "target": {"items": []},
                "override": {"target.items+": ["c"]},
            }
        )
    )
    assert result["target"]["items"] == ["a", "a", "b", "c"]


def test_later_ancestor_replacement_cannot_be_overtaken(tmp_path):
    early = write_config(
        tmp_path, "early.yaml", {"override": {"target.value": "early"}}
    )
    late = write_config(
        tmp_path, "late.yaml", {"override": {"target": {"value": "late"}}}
    )
    with pytest.warns(FutureWarning, match="early.yaml.*target.value"):
        result = load_config(yaml.safe_dump({"include": [early, late]}))
    assert result == {"target": {"value": "late"}}


@pytest.mark.parametrize("strict", ["warn", "error"])
def test_deferred_collection_keeps_declaring_strictness_and_source(tmp_path, strict):
    child = write_config(
        tmp_path,
        "child.yaml",
        {
            "__meta__": {"kind": "fragment", "strict": strict},
            "override": {"missing.items-": ["a"]},
        },
    )
    text = yaml.safe_dump(
        {
            "__meta__": {"strict": "error" if strict == "warn" else "warn"},
            "include": child,
        }
    )
    if strict == "warn":
        with pytest.warns(UserWarning, match="child.yaml.*missing.items"):
            assert load_config(text) == {}
    else:
        with pytest.raises(ConfigPathError, match="child.yaml.*missing.items"):
            load_config(text)


@pytest.mark.parametrize("inline", [False, True])
def test_unresolved_assignment_warns_at_final_boundary(tmp_path, inline):
    path = write_config(tmp_path, "optional.yaml", {"override": {"missing.value": 1}})
    with pytest.warns(FutureWarning, match="optional.yaml.*optional_paths"):
        result = (
            load_config(f"value: !include {path}") if inline else load_config_file(path)
        )
    assert result == ({"value": {}} if inline else {})


@pytest.mark.parametrize("operation", ["=", "+", "-", "remove"])
def test_optional_path_only_skips_absent_target(operation):
    path = "missing.items"
    content = {
        "__meta__": {"optional_paths": [path]},
        "override": {path + (operation if operation in ("+", "-") else ""): [1]},
    }
    if operation == "remove":
        content.pop("override")
        content["remove"] = [path]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_config(yaml.safe_dump(content)) == {}
    content["missing"] = 3
    with pytest.raises(ConfigTypeError):
        load_config(yaml.safe_dump(content))


def test_optional_assignment_still_applies_when_parent_appears(tmp_path):
    fragment = write_config(
        tmp_path,
        "fragment.yaml",
        {
            "__meta__": {"kind": "fragment", "optional_paths": ["target.value"]},
            "override": {"target.value": 7},
        },
    )
    assert load_config(yaml.safe_dump({"include": fragment, "target": {}})) == {
        "target": {"value": 7},
    }


@pytest.mark.parametrize("optional", ["x", [None], [""], ["x..y"]])
def test_optional_paths_validate_metadata(optional):
    with pytest.raises(ConfigValidationError, match="optional_paths"):
        load_config(yaml.safe_dump({"__meta__": {"optional_paths": optional}}))


def test_optional_paths_require_a_local_directive():
    with pytest.raises(ConfigValidationError, match="do not match local"):
        load_config("__meta__: {optional_paths: [typo]}")


def test_named_edits_cannot_be_optional_or_overtake_pending_operations(tmp_path):
    with pytest.raises(ConfigValidationError, match="cannot be optional"):
        load_config("""
__meta__: {optional_paths: [steps]}
override:
  steps~: {remove: {name: one}}
""")
    child = write_config(
        tmp_path,
        "child.yaml",
        {
            "override": {
                "missing.steps+": [{"name": "one"}],
                "missing.steps~": {"remove": {"name": "one"}},
            },
        },
    )
    with pytest.raises(ConfigPathError, match="blocked by.*unresolved"):
        load_config(yaml.safe_dump({"include": child}))


def test_pending_removal_retains_operation_when_retried():
    from spine.config.operations import apply_overrides_and_removals

    config, pending = apply_overrides_and_removals(
        {}, {"target.value": 7}, ["target.value"], "error", "append"
    )
    assert isinstance(pending, list)
    config["target"] = {}
    resolved, remaining = apply_overrides_and_removals(
        config, pending, [], "error", "append", defer_missing=False
    )
    assert resolved == {"target": {}}
    assert remaining == []


@pytest.mark.parametrize("path", ["missing.child", "missing"])
def test_explicit_removal_warns_for_missing_parents_and_leaves(path):
    with pytest.warns(UserWarning, match=path):
        assert (
            load_config(
                yaml.safe_dump(
                    {
                        "__meta__": {"strict": "warn"},
                        "remove": [path],
                    }
                )
            )
            == {}
        )
