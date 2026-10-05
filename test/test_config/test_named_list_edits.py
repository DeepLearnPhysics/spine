"""Named-list edits and their execution order across config composition."""

from copy import deepcopy

import pytest
import yaml

from spine.config import apply_overrides, load_config, load_config_file
from spine.config.errors import ConfigOperationError, ConfigPathError, ConfigTypeError
from spine.config.operations import apply_named_list_edits, apply_overrides_and_removals


@pytest.fixture
def config():
    """Provide a generic named list with nested values to preserve."""
    return {
        "pipeline": {
            "steps": [
                {"name": "first", "config": {"gain": 1, "keep": True}},
                {"name": "last", "config": {"values": [1, 2], "enabled": True}},
            ]
        },
        "unrelated": {"value": 42},
    }


def write_config(tmp_path, name, content):
    """Write a fragment without unrelated metadata warnings."""
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump({"__meta__": {"kind": "fragment"}, **content}, sort_keys=False)
    )
    return str(path)


def insert(name, **anchor):
    """Build an insertion operation with a single named anchor."""
    return {"insert": {**anchor, "value": {"name": name}}}


def test_edits_preserve_unmodified_content_and_do_not_alias_inputs(config):
    """Updates merge mappings, replace lists/nulls, and retain untouched fields."""
    original = deepcopy(config)
    edits = [
        insert("middle", before="last"),
        {"update": {"name": "middle", "changes": {"config": {"gain": 2}}}},
        {
            "update": {
                "name": "last",
                "changes": {"config": {"values": [3], "enabled": None}},
            }
        },
    ]
    edits_copy = deepcopy(edits)
    old_steps = config["pipeline"]["steps"]
    result = apply_named_list_edits(config, "pipeline.steps", edits)
    assert result is config
    assert result["pipeline"]["steps"] == [
        original["pipeline"]["steps"][0],
        {"name": "middle", "config": {"gain": 2}},
        {"name": "last", "config": {"values": [3], "enabled": None}},
    ]
    assert old_steps == original["pipeline"]["steps"]
    assert result["unrelated"] == original["unrelated"]
    result["pipeline"]["steps"][1]["config"]["gain"] = 99
    result["pipeline"]["steps"][2]["config"]["values"].append(4)
    assert edits == edits_copy


def test_immediate_anchors_removal_and_reuse(config):
    """Repeated anchors use the current list; removed names may be reused."""
    apply_named_list_edits(
        config,
        "pipeline.steps",
        [
            insert("a", after="first"),
            insert("b", after="first"),
            {"remove": {"name": "a"}},
            insert("a", before="last"),
            insert("c", after="a"),
        ],
    )
    assert [entry["name"] for entry in config["pipeline"]["steps"]] == [
        "first",
        "b",
        "a",
        "c",
        "last",
    ]


def test_failed_sequence_does_not_partially_apply(config):
    """A failing later operation does not leave an earlier insertion behind."""
    original = deepcopy(config)
    with pytest.raises(ConfigPathError, match="edit 2.*pipeline.steps.*missing"):
        apply_named_list_edits(
            config,
            "pipeline.steps",
            [
                insert("middle", before="last"),
                {"remove": {"name": "missing"}},
            ],
        )
    assert config == original


@pytest.mark.parametrize(
    "edits",
    [
        None,
        [],
        "insert",
        [None],
        [{}],
        [{"unknown": {}}],
        [{"remove": "first"}],
        [{"insert": {}, "remove": {}}],
        insert("new"),
        insert("new", before="first", after="last"),
        {"insert": {"before": "first", "value": {}}},
        {"insert": {"before": "first", "value": {"name": 1}}},
        {"insert": {"before": "first", "value": {"name": " "}}},
        {"insert": {"before": "first", "value": {"name": "new"}, "typo": True}},
        insert("first", before="last"),
        insert("new", before=0),
        {"update": {"name": "first"}},
        {"update": {"name": "first", "changes": []}},
        {"update": {"name": "first", "changes": {"name": "renamed"}}},
        {"remove": {"name": "first", "typo": True}},
        {"remove": {"name": None}},
    ],
)
def test_malformed_edits_fail(config, edits):
    """Malformed operations fail explicitly without changing the config."""
    original = deepcopy(config)
    with pytest.raises(ConfigOperationError):
        apply_named_list_edits(config, "pipeline.steps", edits)
    assert config == original


@pytest.mark.parametrize(
    "entries",
    [
        ["first"],
        [{}],
        [{"name": ""}],
        [{"name": []}],
        [{"name": "same"}, {"name": "same"}],
    ],
)
def test_invalid_named_list_fails(entries):
    """Even untouched entries must have unique nonempty string names."""
    with pytest.raises(ConfigOperationError):
        apply_named_list_edits(
            {"steps": entries}, "steps", {"remove": {"name": "first"}}
        )


@pytest.mark.parametrize(
    ("cfg", "path", "error"),
    [
        ({}, "missing.steps", ConfigPathError),
        ({"pipeline": {}}, "pipeline.steps", ConfigPathError),
        ({"pipeline": []}, "pipeline.steps", ConfigTypeError),
        ({"steps": {}}, "steps", ConfigTypeError),
        ({}, "", ConfigPathError),
        ({}, "a..b", ConfigPathError),
    ],
)
def test_invalid_paths_fail(cfg, path, error):
    """Missing paths and wrong collection types are never silently created."""
    with pytest.raises(error):
        apply_named_list_edits(cfg, path, insert("new", before="first"))


@pytest.mark.parametrize(
    "edit",
    [
        insert("new", before="missing"),
        {"update": {"name": "missing", "changes": {}}},
        {"remove": {"name": "missing"}},
    ],
)
def test_missing_names_fail_even_in_warn_mode(config, edit):
    """Named edits neither warn-and-skip nor propagate unresolved targets."""
    with pytest.raises(ConfigPathError, match="missing"):
        apply_overrides_and_removals(
            config, {"pipeline.steps~": edit}, [], "warn", "append"
        )


@pytest.mark.parametrize("entrypoint", ["file", "string", "inline", "cli"])
def test_all_entrypoints_apply_named_edits(tmp_path, config, entrypoint):
    """File, string, YAML tag, and CLI paths share the same edit semantics."""
    edit = insert("middle", before="last")
    content = {**config, "override": {"pipeline.steps~": edit}}
    path = write_config(tmp_path, "base.yaml", content)
    if entrypoint == "file":
        result = load_config_file(path)
    elif entrypoint == "string":
        result = load_config(yaml.safe_dump(content))
    elif entrypoint == "inline":
        result = load_config("wrapped: !include base.yaml", root_dir=str(tmp_path))[
            "wrapped"
        ]
    else:
        result = apply_overrides(
            config, ["pipeline.steps~=" + yaml.safe_dump(edit, default_flow_style=True)]
        )
    assert [entry["name"] for entry in result["pipeline"]["steps"]] == [
        "first",
        "middle",
        "last",
    ]
    assert "override" not in result
    assert "steps~" not in result["pipeline"]


def test_nested_resolution_matches_loading_resolved_intermediate(tmp_path, config):
    """A -> B -> C is equivalent to loading C with an already resolved B."""
    write_config(tmp_path, "a.yaml", config)
    b = write_config(
        tmp_path,
        "b.yaml",
        {
            "include": "a.yaml",
            "override": {"pipeline.steps~": insert("middle", before="last")},
        },
    )
    c_content = {
        "include": "b.yaml",
        "override": {
            "pipeline.steps~": [
                {"update": {"name": "middle", "changes": {"config": {"gain": 1.1}}}},
                {"remove": {"name": "first"}},
            ]
        },
    }
    c = write_config(tmp_path, "c.yaml", c_content)
    nested = load_config_file(c)
    resolved_b = load_config_file(b)
    write_config(tmp_path, "resolved_b.yaml", resolved_b)
    c_content["include"] = "resolved_b.yaml"
    flattened = load_config_file(write_config(tmp_path, "flat.yaml", c_content))
    assert nested == flattened
    assert nested["pipeline"]["steps"] == [
        {"name": "middle", "config": {"gain": 1.1}},
        config["pipeline"]["steps"][1],
    ]


def test_sibling_modifiers_and_parent_edits_execute_once_in_order(tmp_path, config):
    """Modifiers on the same path are not overwritten, deferred, or repeated."""
    write_config(tmp_path, "a.yaml", config)
    write_config(
        tmp_path,
        "b.yaml",
        {"override": {"pipeline.steps~": insert("middle", after="first")}},
    )
    write_config(
        tmp_path,
        "c.yaml",
        {"override": {"pipeline.steps~": insert("next", after="middle")}},
    )
    path = write_config(
        tmp_path,
        "bundle.yaml",
        {
            "include": ["a.yaml", "b.yaml", "c.yaml"],
            "override": {"pipeline.steps~": {"remove": {"name": "last"}}},
        },
    )
    result = load_config_file(path)
    assert [entry["name"] for entry in result["pipeline"]["steps"]] == [
        "first",
        "middle",
        "next",
    ]


def test_missing_target_does_not_wait_for_later_include(tmp_path, config):
    """An edit cannot refer forward to a later sibling's configuration."""
    write_config(tmp_path, "a.yaml", config)
    write_config(
        tmp_path,
        "mod.yaml",
        {"override": {"pipeline.steps~": insert("middle", before="last")}},
    )
    path = write_config(tmp_path, "bundle.yaml", {"include": ["mod.yaml", "a.yaml"]})
    with pytest.raises(ConfigPathError, match="pipeline.steps"):
        load_config_file(path)


def test_later_include_replacement_wins_over_earlier_edit(tmp_path, config):
    """Edits are applied at include time, not replayed after all composition."""
    write_config(tmp_path, "a.yaml", config)
    write_config(
        tmp_path,
        "mod.yaml",
        {"override": {"pipeline.steps~": insert("middle", before="last")}},
    )
    write_config(
        tmp_path, "replacement.yaml", {"pipeline": {"steps": [{"name": "replacement"}]}}
    )
    path = write_config(
        tmp_path, "bundle.yaml", {"include": ["a.yaml", "mod.yaml", "replacement.yaml"]}
    )
    assert load_config_file(path)["pipeline"]["steps"] == [{"name": "replacement"}]


def test_legacy_collection_operations_mix_with_named_edits(config):
    """Append, named edit, and value removal execute in mapping order."""
    content = {
        **config,
        "override": {
            "pipeline.steps+": [{"name": "appended"}],
            "pipeline.steps~": [
                insert("middle", before="appended"),
                {"update": {"name": "first", "changes": {"config": {"gain": 2}}}},
            ],
            "pipeline.steps-": [{"name": "appended"}],
        },
    }
    result = load_config(yaml.safe_dump(content, sort_keys=False))
    assert [entry["name"] for entry in result["pipeline"]["steps"]] == [
        "first",
        "last",
        "middle",
    ]
    assert result["pipeline"]["steps"][0]["config"] == {"gain": 2, "keep": True}


def test_parent_values_precede_own_edits_and_cli_runs_last(tmp_path, config):
    """Local configuration is available to edits; CLI overrides remain last."""
    write_config(tmp_path, "base.yaml", config)
    replacement = [{"name": "local"}]
    path = write_config(
        tmp_path,
        "bundle.yaml",
        {
            "include": "base.yaml",
            "pipeline": {"steps": replacement},
            "override": {"pipeline.steps~": insert("added", before="local")},
        },
    )
    loaded = load_config_file(path)
    assert [entry["name"] for entry in loaded["pipeline"]["steps"]] == [
        "added",
        "local",
    ]
    result = apply_overrides(
        loaded,
        [
            "pipeline.steps~={update: {name: added, changes: {config: {gain: 2}}}}",
            "pipeline.steps~={remove: {name: local}}",
        ],
    )
    assert result["pipeline"]["steps"] == [{"name": "added", "config": {"gain": 2}}]


def test_repeated_before_anchor_retains_edit_order(config):
    """Insertions before the same anchor retain their sequence order."""
    apply_named_list_edits(
        config,
        "pipeline.steps",
        [
            insert("a", before="last"),
            insert("b", before="last"),
        ],
    )
    assert [entry["name"] for entry in config["pipeline"]["steps"]] == [
        "first",
        "a",
        "b",
        "last",
    ]


def test_removed_target_cannot_be_used_later(config):
    """Each operation resolves against the result of the previous operation."""
    with pytest.raises(ConfigPathError, match="target 'first'"):
        apply_named_list_edits(
            config,
            "pipeline.steps",
            [
                {"remove": {"name": "first"}},
                insert("new", after="first"),
            ],
        )
