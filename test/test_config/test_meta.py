"""Tests for spine.config metadata helpers."""

import pytest

from spine.config.api import (
    META_COMPATIBLE_WITH,
    META_KIND,
    META_LIST_APPEND,
    META_STRICT,
)
from spine.config.errors import ConfigCycleError, ConfigValidationError
from spine.config.meta import (
    _compare_versions,
    _parse_version_constraint,
    check_compatibility,
    extract_metadata,
)


class TestCompareVersions:
    """Tests for version comparison helper."""

    @pytest.mark.parametrize(
        ("actual", "operator", "required", "expected"),
        [
            ("240719", "==", "240719", True),
            ("240720", ">=", "240719", True),
            ("240718", "<=", "240719", True),
            ("240720", ">", "240719", True),
            ("240718", "<", "240719", True),
            ("240718", "!=", "240719", True),
        ],
    )
    def test_known_operators(self, actual, operator, required, expected):
        """Test supported version operators."""
        assert _compare_versions(actual, operator, required) is expected

    def test_unknown_operator_warns_and_defaults_to_equals(self):
        """Test unknown operators emit a warning and fall back to equality."""
        with pytest.warns(UserWarning, match="Unknown version operator"):
            assert _compare_versions("240719", "~", "240719") is True

    @pytest.mark.parametrize("actual", ["", "24071", "2407190", "24a719"])
    def test_invalid_actual_version_raises(self, actual):
        """Test actual versions must use the expected six digit format."""
        with pytest.raises(ValueError, match="Invalid version format"):
            _compare_versions(actual, "==", "240719")

    @pytest.mark.parametrize("required", ["", "24071", "2407190", "24a719"])
    def test_invalid_required_version_raises(self, required):
        """Test required versions must use the expected six digit format."""
        with pytest.raises(ValueError, match="Invalid version format"):
            _compare_versions("240719", "==", required)


class TestParseVersionConstraint:
    """Tests for version constraint parsing."""

    @pytest.mark.parametrize(
        ("constraint", "expected"),
        [
            (">=240719", (">=", "240719")),
            ("<=240719", ("<=", "240719")),
            ("==240719", ("==", "240719")),
            ("!=240719", ("!=", "240719")),
            (">240719", (">", "240719")),
            ("<240719", ("<", "240719")),
            ("240719", ("==", "240719")),
            ("  >=240719  ", (">=", "240719")),
        ],
    )
    def test_parse_version_constraint(self, constraint, expected):
        """Test operator parsing and default equality behavior."""
        assert _parse_version_constraint(constraint) == expected


class TestExtractMetadata:
    """Tests for metadata extraction and validation."""

    def test_extract_metadata_defaults_for_bundle(self):
        """Test bundle configs use bundle defaults."""
        metadata = extract_metadata({})

        assert metadata[META_KIND] == "bundle"
        assert metadata[META_STRICT] == "error"
        assert metadata[META_LIST_APPEND] == "append"

    def test_extract_metadata_defaults_for_mod(self):
        """Test mod configs default to warning strictness."""
        metadata = extract_metadata({"__meta__": {META_KIND: "mod"}})

        assert metadata[META_KIND] == "mod"
        assert metadata[META_STRICT] == "warn"

    def test_extract_metadata_warns_and_normalizes_invalid_values(self):
        """Test invalid metadata values warn and fall back to defaults."""
        config = {
            "__meta__": {
                META_KIND: "invalid-kind",
                META_STRICT: "invalid-strict",
                META_LIST_APPEND: "invalid-append",
                "custom": "value",
            }
        }

        with pytest.warns(UserWarning, match="Invalid __meta__\\.kind"):
            with pytest.warns(UserWarning, match="Invalid __meta__\\.strict"):
                with pytest.warns(UserWarning, match="Invalid __meta__\\.list_append"):
                    metadata = extract_metadata(config, cfg_path="/tmp/config.yaml")

        assert metadata[META_KIND] == "bundle"
        assert metadata[META_STRICT] == "error"
        assert metadata[META_LIST_APPEND] == "append"
        assert metadata["custom"] == "value"
        assert metadata["_file_path"] == "/tmp/config.yaml"


class TestCheckCompatibility:
    """Tests for compatibility validation logic."""

    def test_no_compatibility_requirements_is_noop(self):
        """Test included configs without compatibility metadata are accepted."""
        check_compatibility({}, {}, "included.yaml")

    def test_dict_constraints_pass_with_components(self):
        """Test version constraints can be satisfied from parent components."""
        parent_meta = {
            "_file_path": "/configs/parent.yaml",
            "components": {"base": "240719", "io": "240812"},
        }
        included_meta = {META_COMPATIBLE_WITH: {"base": "==240719", "io": ">=240800"}}

        check_compatibility(parent_meta, included_meta, "/configs/included.yaml")

    def test_dict_constraints_fall_back_to_parent_version(self):
        """Test version checks fall back to the parent version when needed."""
        parent_meta = {"_file_path": "/configs/parent.yaml", "version": "240719"}
        included_meta = {META_COMPATIBLE_WITH: {"base": "240719"}}

        check_compatibility(parent_meta, included_meta, "/configs/included.yaml")

    def test_dict_constraints_raise_for_missing_version(self):
        """Test missing parent versions are reported as compatibility failures."""
        parent_meta = {"_file_path": "/configs/parent.yaml", "components": {}}
        included_meta = {META_COMPATIBLE_WITH: {"base": "240719"}}

        with pytest.raises(ConfigValidationError, match="parent has no version"):
            check_compatibility(parent_meta, included_meta, "/configs/included.yaml")

    def test_dict_constraints_raise_for_version_mismatch(self):
        """Test incompatible version constraints raise validation errors."""
        parent_meta = {
            "_file_path": "/configs/parent.yaml",
            "components": {"base": "240718"},
        }
        included_meta = {META_COMPATIBLE_WITH: {"base": ">=240719"}}

        with pytest.raises(ConfigValidationError, match="does not satisfy >=240719"):
            check_compatibility(parent_meta, included_meta, "/configs/included.yaml")

    def test_legacy_string_compatibility_accepts_parent_identifiers(self):
        """Test legacy string compatibility matches version-qualified identifiers."""
        parent_meta = {
            META_KIND: "bundle",
            "extends": "icarus",
            "version": "240719",
            "_file_path": "/configs/parent.yaml",
        }
        included_meta = {META_COMPATIBLE_WITH: "icarus_240719"}

        check_compatibility(parent_meta, included_meta, "/configs/included.yaml")

    def test_legacy_list_compatibility_accepts_parent_extends_list(self):
        """Test legacy list compatibility matches parent extends values."""
        parent_meta = {
            META_KIND: "bundle",
            "extends": ["icarus", "sbnd"],
            "_file_path": "/configs/parent.yaml",
        }
        included_meta = {META_COMPATIBLE_WITH: ["sbnd", "dune"]}

        check_compatibility(parent_meta, included_meta, "/configs/included.yaml")

    def test_legacy_compatibility_mismatch_raises(self):
        """Test mismatched legacy compatibility metadata raises an error."""
        parent_meta = {META_KIND: "bundle", "_file_path": "/configs/parent.yaml"}
        included_meta = {META_COMPATIBLE_WITH: ["mod"]}

        with pytest.raises(ConfigValidationError, match="declares compatible_with"):
            check_compatibility(parent_meta, included_meta, "/configs/included.yaml")

    def test_conflicting_modifiers_warn(self):
        """Test overlapping modifier extends values emit a warning."""
        parent_meta = {
            META_KIND: "mod",
            "extends": ["icarus", "base"],
            "_file_path": "/configs/parent.yaml",
        }
        included_meta = {
            META_COMPATIBLE_WITH: ["mod"],
            "extends": ["icarus", "analysis"],
        }

        with pytest.warns(UserWarning, match="Potential modifier conflict"):
            check_compatibility(parent_meta, included_meta, "/configs/included.yaml")


class TestConfigErrors:
    """Tests for configuration exception helpers."""

    def test_config_cycle_error_keeps_path_and_message(self):
        """Test cycle errors retain the full cycle path for debugging."""
        cycle_path = ["a.yaml", "b.yaml", "a.yaml"]

        error = ConfigCycleError(cycle_path)

        assert error.cycle_path == cycle_path
        assert str(error) == "Circular include detected: a.yaml -> b.yaml -> a.yaml"
