"""Tests for spine.config.operations helpers."""

import pytest

from spine.config.errors import ConfigOperationError, ConfigPathError, ConfigTypeError
from spine.config.operations import (
    _apply_overrides_and_removals,
    apply_collection_operation,
    deep_merge,
    expand_env_vars,
    extract_includes_and_overrides,
    parse_value,
    set_nested_value,
)


class TestDeepMerge:
    """Tests for recursive dictionary merging."""

    def test_deep_merge_recursively_merges_without_mutating_inputs(self):
        """Test nested dictionaries are merged into a new object."""
        base = {"model": {"layers": 3, "width": 64}, "name": "base"}
        override = {"model": {"width": 128}, "enabled": True}

        merged = deep_merge(base, override)

        assert merged == {
            "model": {"layers": 3, "width": 128},
            "name": "base",
            "enabled": True,
        }
        assert base == {"model": {"layers": 3, "width": 64}, "name": "base"}
        assert override == {"model": {"width": 128}, "enabled": True}


class TestParseValue:
    """Tests for value parsing helper."""

    def test_parse_value_passthrough_for_non_strings(self):
        """Test non-string values are returned unchanged."""
        value = {"a": 1}
        assert parse_value(value) is value

    def test_parse_value_preserves_empty_strings(self):
        """Test empty strings are not YAML parsed."""
        assert parse_value("   ") == "   "

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("[1, 2]", [1, 2]), ("true", True), ("3.5", 3.5), ("a: 1", {"a": 1})],
    )
    def test_parse_value_yaml_parses_valid_strings(self, raw, expected):
        """Test YAML-compatible string values are parsed."""
        assert parse_value(raw) == expected

    def test_parse_value_returns_original_on_yaml_error(self):
        """Test invalid YAML content is returned unchanged."""
        raw = "[1, 2"
        assert parse_value(raw) == raw


class TestExpandEnvVars:
    """Tests for environment variable expansion."""

    def test_expand_env_vars_recurses_through_nested_values(self, monkeypatch):
        """Test strings inside dicts and lists have variables expanded."""
        monkeypatch.setenv("SPINE_DATA_ROOT", "/tmp/data")

        value = {
            "path": "$SPINE_DATA_ROOT/file.root",
            "items": ["$SPINE_DATA_ROOT/a", {"inner": "$SPINE_DATA_ROOT/b"}],
            "count": 3,
        }

        expanded = expand_env_vars(value)

        assert expanded == {
            "path": "/tmp/data/file.root",
            "items": ["/tmp/data/a", {"inner": "/tmp/data/b"}],
            "count": 3,
        }


class TestApplyCollectionOperation:
    """Tests for collection operation helper."""

    def test_append_creates_missing_list(self):
        """Test append creates a new list when the target key does not exist."""
        config = {"io": {}}

        result = apply_collection_operation(config, "io.file_keys", "a", "+")

        assert result["io"]["file_keys"] == ["a"]

    def test_append_unique_skips_duplicates(self):
        """Test unique append mode avoids duplicating existing values."""
        config = {"io": {"file_keys": ["a"]}}

        result = apply_collection_operation(
            config, "io.file_keys", ["a", "b"], "+", list_append_mode="unique"
        )

        assert result["io"]["file_keys"] == ["a", "b"]

    def test_remove_list_drops_all_occurrences(self):
        """Test list removals remove all matching values."""
        config = {"io": {"file_keys": ["a", "b", "a"]}}

        result = apply_collection_operation(config, "io.file_keys", "a", "-")

        assert result["io"]["file_keys"] == ["b"]

    def test_missing_path_warns_in_warn_mode(self):
        """Test missing paths warn instead of raising when strict is warn."""
        config = {}

        with pytest.warns(UserWarning, match="path does not exist"):
            result = apply_collection_operation(
                config, "io.file_keys", "a", "+", strict="warn"
            )

        assert result == {}

    def test_missing_path_raises_in_error_mode(self):
        """Test missing paths raise in strict error mode."""
        with pytest.raises(ConfigPathError, match="path does not exist"):
            apply_collection_operation({}, "io.file_keys", "a", "+", strict="error")

    def test_missing_remove_key_warns_in_warn_mode(self):
        """Test removing a missing key warns in warn mode."""
        config = {"io": {}}

        with pytest.warns(UserWarning, match="key does not exist"):
            result = apply_collection_operation(
                config, "io.file_keys", "a", "-", strict="warn"
            )

        assert result == {"io": {}}

    def test_missing_remove_key_raises_in_error_mode(self):
        """Test removing a missing key raises in strict error mode."""
        with pytest.raises(ConfigPathError, match="key does not exist"):
            apply_collection_operation(
                {"io": {}}, "io.file_keys", "a", "-", strict="error"
            )

    def test_non_dict_parent_raises_type_error(self):
        """Test traversal through non-dict parents raises a type error."""
        with pytest.raises(ConfigTypeError, match="is not a dictionary"):
            apply_collection_operation({"io": []}, "io.file_keys", "a", "+")

    def test_invalid_operation_on_list_raises(self):
        """Test unsupported list operations raise a configuration error."""
        with pytest.raises(ConfigOperationError, match="Invalid collection operation"):
            apply_collection_operation(
                {"io": {"file_keys": []}}, "io.file_keys", "a", "*"
            )

    def test_dict_removal_warns_for_missing_key(self):
        """Test missing dict removals warn in warn mode."""
        config = {"io": {"options": {"keep": 1}}}

        with pytest.warns(UserWarning, match="not found in 'io.options'"):
            result = apply_collection_operation(
                config, "io.options", ["missing"], "-", strict="warn"
            )

        assert result["io"]["options"] == {"keep": 1}

    def test_dict_removal_deletes_existing_key(self):
        """Test dictionary removals delete existing keys."""
        config = {"io": {"options": {"keep": 1, "drop": 2}}}

        result = apply_collection_operation(config, "io.options", ["drop"], "-")

        assert result["io"]["options"] == {"keep": 1}

    def test_append_to_dict_raises(self):
        """Test dict append operations are rejected."""
        with pytest.raises(ConfigOperationError, match="not supported for dicts"):
            apply_collection_operation({"io": {"options": {}}}, "io.options", "a", "+")

    def test_invalid_operation_on_dict_raises(self):
        """Test unsupported dict operations raise a configuration error."""
        with pytest.raises(ConfigOperationError, match="Invalid collection operation"):
            apply_collection_operation({"io": {"options": {}}}, "io.options", "a", "*")

    def test_non_collection_target_raises(self):
        """Test scalar targets are rejected."""
        with pytest.raises(ConfigTypeError, match="not a list or dict"):
            apply_collection_operation({"io": {"count": 1}}, "io.count", "a", "+")


class TestSetNestedValue:
    """Tests for nested set/delete helper."""

    def test_set_nested_value_creates_parents(self):
        """Test setting values creates intermediate dictionaries as needed."""
        config, applied = set_nested_value({}, "io.reader.path", "file.root")

        assert applied is True
        assert config == {"io": {"reader": {"path": "file.root"}}}

    def test_set_nested_value_only_if_exists_skips_missing_path(self):
        """Test only_if_exists leaves config unchanged when path is missing."""
        config, applied = set_nested_value(
            {}, "io.reader.path", "file.root", only_if_exists=True
        )

        assert applied is False
        assert config == {}

    def test_delete_missing_parent_raises_in_error_mode(self):
        """Test deleting through a missing path raises in strict error mode."""
        with pytest.raises(ConfigPathError, match="path 'io' does not exist"):
            set_nested_value({}, "io.reader.path", None, delete=True)

    def test_delete_missing_parent_returns_false_in_warn_mode(self):
        """Test missing delete paths are ignored in warn mode."""
        config, applied = set_nested_value(
            {}, "io.reader.path", None, delete=True, strict="warn"
        )

        assert applied is False
        assert config == {}

    def test_set_nested_value_rejects_non_dict_parent(self):
        """Test traversing through a scalar parent raises a type error."""
        with pytest.raises(ConfigTypeError, match="is not a dictionary"):
            set_nested_value({"io": 1}, "io.reader.path", "file.root")

    def test_delete_existing_value(self):
        """Test deleting an existing final key succeeds."""
        config, applied = set_nested_value(
            {"io": {"path": "file.root"}}, "io.path", None, delete=True
        )

        assert applied is True
        assert config == {"io": {}}

    def test_delete_missing_key_warns(self):
        """Test deleting a missing final key warns in warn mode."""
        with pytest.warns(UserWarning, match="not found, skipping deletion"):
            config, applied = set_nested_value(
                {"io": {}}, "io.path", None, delete=True, strict="warn"
            )

        assert applied is False
        assert config == {"io": {}}

    def test_delete_missing_key_raises_in_error_mode(self):
        """Test deleting a missing final key raises in strict error mode."""
        with pytest.raises(ConfigPathError, match="key does not exist"):
            set_nested_value({"io": {}}, "io.path", None, delete=True, strict="error")


class TestExtractIncludesAndOverrides:
    """Tests for directive extraction."""

    def test_non_dict_config_is_passthrough(self):
        """Test non-dictionary configs are returned untouched."""
        config = ["a", "b"]

        assert extract_includes_and_overrides(config) == ([], {}, [], config)

    def test_extracts_directives_and_cleaned_config(self):
        """Test include, override, and remove directives are separated cleanly."""
        config = {
            "include": ["base.yaml", "io.yaml"],
            "override": {"io.path": "file.root"},
            "remove": ["analysis.old", "analysis.legacy"],
            "io": {"mode": "reader"},
        }

        includes, overrides, removals, cleaned = extract_includes_and_overrides(config)

        assert includes == ["base.yaml", "io.yaml"]
        assert overrides == {"io.path": "file.root"}
        assert removals == ["analysis.old", "analysis.legacy"]
        assert cleaned == {"io": {"mode": "reader"}}

    def test_extracts_single_string_directives(self):
        """Test single include/remove strings are normalized to one-item lists."""
        includes, overrides, removals, cleaned = extract_includes_and_overrides(
            {"include": "base.yaml", "remove": "analysis.old", "value": 1}
        )

        assert includes == ["base.yaml"]
        assert overrides == {}
        assert removals == ["analysis.old"]
        assert cleaned == {"value": 1}

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("include", 1, "'include' must be a string or list of strings"),
            ("override", "bad", "'override' must be a dictionary"),
            ("remove", 1, "'remove' must be a string or list of strings"),
        ],
    )
    def test_invalid_directive_types_raise(self, field, value, match):
        """Test directive type validation."""
        with pytest.raises(ConfigOperationError, match=match):
            extract_includes_and_overrides({field: value})


class TestApplyOverridesAndRemovals:
    """Tests for override/removal orchestration."""

    def test_apply_overrides_and_removals_tracks_unapplied_paths(self):
        """Test missing override paths are returned for propagation."""
        config = {"io": {"path": "old.root", "old": 1}, "list": ["a"]}
        overrides = {"io.path": "new.root", "missing.value": "1", "list+": "b"}

        updated, unapplied = _apply_overrides_and_removals(
            config,
            overrides,
            ["io.old"],
            strict="error",
            list_append_mode="append",
        )

        assert updated == {"io": {"path": "new.root"}, "list": ["a", "b"]}
        assert unapplied == {"missing.value": "1"}

    def test_apply_overrides_and_removals_propagates_type_errors(self):
        """Test type errors from collection operations are not swallowed."""
        with pytest.raises(ConfigTypeError, match="not a list or dict"):
            _apply_overrides_and_removals(
                {"io": {"count": 1}},
                {"io.count+": "2"},
                [],
                strict="error",
                list_append_mode="append",
            )

    def test_apply_overrides_and_removals_tracks_unapplied_collection_paths(self):
        """Test collection overrides on missing paths are returned as unapplied."""
        updated, unapplied = _apply_overrides_and_removals(
            {},
            {"io.file_keys+": "a"},
            [],
            strict="error",
            list_append_mode="append",
        )

        assert updated == {}
        assert unapplied == {"io.file_keys+": "a"}

    def test_apply_overrides_and_removals_tracks_unapplied_regular_paths(self):
        """Test regular overrides on missing parents are returned as unapplied."""
        updated, unapplied = _apply_overrides_and_removals(
            {},
            {"io.path": "file.root"},
            [],
            strict="error",
            list_append_mode="append",
        )

        assert updated == {}
        assert unapplied == {"io.path": "file.root"}

    def test_apply_overrides_and_removals_reraises_config_path_type_hint(
        self, monkeypatch
    ):
        """Test ConfigPathError messages hinting at type issues are re-raised."""

        def fake_apply_collection_operation(*args, **kwargs):
            raise ConfigPathError("target is not a list")

        monkeypatch.setattr(
            "spine.config.operations.apply_collection_operation",
            fake_apply_collection_operation,
        )

        with pytest.raises(ConfigPathError, match="not a list"):
            _apply_overrides_and_removals(
                {},
                {"io.path+": "value"},
                [],
                strict="error",
                list_append_mode="append",
            )
