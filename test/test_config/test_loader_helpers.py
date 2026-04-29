"""Direct tests for spine.config load/loader helper branches."""

import os

import pytest
import yaml

from spine.config.api import META_KEY, META_LIST_APPEND, META_STRICT
from spine.config.errors import ConfigCycleError, ConfigIncludeError
from spine.config.load import _load_config_recursive, load_config_file
from spine.config.loader import ConfigLoader, DownloadTag, resolve_config_path


class TestLoadHelpers:
    """Tests for low-level recursive load helpers."""

    def test_load_config_recursive_requires_exactly_one_input(self):
        """Test helper rejects both and neither cfg_path/config_string."""
        with pytest.raises(ValueError, match="exactly one"):
            _load_config_recursive()

        with pytest.raises(ValueError, match="exactly one"):
            _load_config_recursive(cfg_path="a.yaml", config_string="base: 1")

    def test_load_config_recursive_missing_file_raises_config_include_error(
        self, tmp_path
    ):
        """Test missing files are wrapped as ConfigIncludeError."""
        missing = tmp_path / "missing.yaml"

        with pytest.raises(ConfigIncludeError, match="Configuration file not found"):
            _load_config_recursive(cfg_path=str(missing))

    def test_load_config_recursive_wraps_loader_errors(self, tmp_path):
        """Test malformed YAML is wrapped with source context."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("foo: [1, 2\n")

        with pytest.raises(ConfigIncludeError, match="Error loading"):
            _load_config_recursive(cfg_path=str(bad_config))

    def test_load_config_recursive_empty_yaml_returns_empty_parts(self, tmp_path):
        """Test empty YAML files return empty config fragments."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")

        result = _load_config_recursive(cfg_path=str(empty))

        assert result == ({}, {}, [], {})

    def test_load_config_file_detects_cycles(self, tmp_path):
        """Test recursive include cycles raise ConfigCycleError."""
        first = tmp_path / "first.yaml"
        second = tmp_path / "second.yaml"
        first.write_text("include: second.yaml\nvalue: 1\n")
        second.write_text("include: first.yaml\nvalue: 2\n")

        with pytest.raises(ConfigCycleError, match="Circular include detected"):
            load_config_file(str(first))

    def test_included_override_propagates_to_main_config(self, tmp_path):
        """Test unapplied overrides from includes are retried after main merge."""
        modifier = tmp_path / "modifier.yaml"
        modifier.write_text("""
override:
  model.name: propagated
""")

        main = tmp_path / "main.yaml"
        main.write_text("""
include: modifier.yaml

model: {}
""")

        cfg = load_config_file(str(main))

        assert cfg["model"]["name"] == "propagated"

    def test_infers_component_version_from_include_directory(self, tmp_path):
        """Test include directories become inferred component names for compatibility."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_cfg = base_dir / "base_config.yaml"
        base_cfg.write_text("""
__meta__:
  version: "240719"

base:
  value: 1
""")

        modifier = tmp_path / "modifier.yaml"
        modifier.write_text("""
__meta__:
  compatible_with:
    base: "240719"

model:
  name: ok
""")

        main = tmp_path / "main.yaml"
        main.write_text("""
include:
  - base/base_config.yaml
  - modifier.yaml
""")

        cfg = load_config_file(str(main))

        assert cfg["base"]["value"] == 1
        assert cfg["model"]["name"] == "ok"

        def test_load_config_recursive_accumulates_explicit_component_metadata(
            self, tmp_path
        ):
            """Test included metadata components are merged into parent metadata."""
            component = tmp_path / "component.yaml"
            component.write_text("""
__meta__:
    components:
        base: "240719"

base:
    value: 1
""")

            main = tmp_path / "main.yaml"
            main.write_text("include: component.yaml\n")

            _config, _overrides, _removals, metadata = _load_config_recursive(
                cfg_path=str(main)
            )

            assert metadata["components"] == {"base": "240719"}

    def test_load_config_recursive_merges_included_component_metadata_via_patch(
        self, tmp_path, monkeypatch
    ):
        """Test parent metadata initializes and updates explicit included components."""
        main = tmp_path / "main.yaml"
        child = tmp_path / "child.yaml"
        main.write_text("include: child.yaml\n")
        child.write_text("value: 1\n")

        original_extract_metadata = __import__(
            "spine.config.load", fromlist=["extract_metadata"]
        ).extract_metadata

        def fake_extract_metadata(config_dict, cfg_path=None):
            if cfg_path and str(cfg_path).endswith("child.yaml"):
                return {
                    META_STRICT: "error",
                    META_LIST_APPEND: "append",
                    "components": {"base": "240719"},
                }
            return original_extract_metadata(config_dict, cfg_path)

        monkeypatch.setattr("spine.config.load.extract_metadata", fake_extract_metadata)

        _config, _overrides, _removals, metadata = _load_config_recursive(
            cfg_path=str(main)
        )

        assert metadata["components"] == {"base": "240719"}

    def test_load_config_string_top_level_collection_override_and_remove(self):
        """Test load_config applies top-level collection operations and removals."""
        from spine.config.load import load_config

        cfg = load_config("""
__meta__:
  version: "240719"

parsers:
  - one
  - two
  - three

override:
  parsers+: four

remove: legacy

legacy: true
""")

        assert cfg["parsers"] == ["one", "two", "three", "four"]
        assert "legacy" not in cfg
        assert META_KEY not in cfg

    def test_load_config_wrapper_applies_collection_remove_and_meta_strip(
        self, monkeypatch
    ):
        """Test load_config wrapper applies deferred operations and strips __meta__."""
        from spine.config.load import load_config

        monkeypatch.setattr(
            "spine.config.load._load_config_recursive",
            lambda **_kwargs: (
                {META_KEY: {"version": "240719"}, "parsers": ["one"], "legacy": True},
                {"parsers+": "two"},
                ["legacy"],
                {META_STRICT: "error", META_LIST_APPEND: "append"},
            ),
        )

        cfg = load_config("ignored")

        assert cfg == {"parsers": ["one", "two"]}

    def test_load_config_file_top_level_remove_and_meta_strip(self, tmp_path):
        """Test load_config_file applies removals and strips __meta__."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
__meta__:
  version: "240719"

base:
  keep: 1
  drop: 2

remove: base.drop
""")

        cfg = load_config_file(str(config_file))

        assert cfg == {"base": {"keep": 1}}

    def test_load_config_file_wrapper_strips_meta(self, tmp_path, monkeypatch):
        """Test load_config_file wrapper strips __meta__ from final configs."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("base: 1\n")

        monkeypatch.setattr(
            "spine.config.load._load_config_recursive",
            lambda **_kwargs: (
                {META_KEY: {"version": "240719"}, "base": {"keep": 1}},
                {},
                [],
                {META_STRICT: "error", META_LIST_APPEND: "append"},
            ),
        )

        cfg = load_config_file(str(config_file))

        assert cfg == {"base": {"keep": 1}}


class TestLoaderHelpers:
    """Tests for ConfigLoader and path resolution helpers."""

    def test_resolve_config_path_missing_absolute_path_raises(self, tmp_path):
        """Test missing absolute paths raise a specific error."""
        missing = tmp_path / "missing.yaml"

        with pytest.raises(ConfigIncludeError, match="Absolute path not found"):
            resolve_config_path(str(missing), str(tmp_path))

    def test_resolve_config_path_missing_without_env_suggests_spine_config_path(
        self, tmp_path, monkeypatch
    ):
        """Test missing files suggest configuring SPINE_CONFIG_PATH when unset."""
        monkeypatch.delenv("SPINE_CONFIG_PATH", raising=False)

        with pytest.raises(ConfigIncludeError, match="Tip: Set SPINE_CONFIG_PATH"):
            resolve_config_path("missing.yaml", str(tmp_path))

    def test_resolve_config_path_adds_relative_yaml_extension(self, tmp_path):
        """Test relative paths resolve with an added .yaml extension."""
        config = tmp_path / "base.yaml"
        config.write_text("base: true\n")

        resolved = resolve_config_path("base", str(tmp_path))

        assert resolved == str(config)

    def test_config_loader_uses_stream_directory_by_default(self, tmp_path):
        """Test file-stream loaders default their root to the file directory."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("value: 1\n")

        with open(config_file, "r", encoding="utf-8") as stream:
            loader = ConfigLoader(stream)

        assert loader._root == str(tmp_path)

    def test_config_loader_uses_cwd_for_string_input(self, tmp_path, monkeypatch):
        """Test string-input loaders default their root to the current working directory."""
        monkeypatch.chdir(tmp_path)

        loader = ConfigLoader("value: 1\n")

        assert loader._root == str(tmp_path)

    def test_download_tag_init_stores_value(self):
        """Test DownloadTag preserves unresolved values."""
        tag = DownloadTag({"url": "https://example.com/model.ckpt"})

        assert tag.value == {"url": "https://example.com/model.ckpt"}

    def test_loader_download_returns_unresolved_scalar_tag_when_disabled(self):
        """Test disabled downloads preserve scalar tag values."""
        loader = ConfigLoader("", download=False)
        node = yaml.compose("!download https://example.com/model.ckpt")

        result = loader.download(node)

        assert isinstance(result, DownloadTag)
        assert result.value == "https://example.com/model.ckpt"

    def test_loader_download_returns_unresolved_mapping_tag_when_disabled(self):
        """Test disabled downloads preserve mapping tag values."""
        loader = ConfigLoader("", download=False)
        node = yaml.compose(
            "!download\nurl: https://example.com/model.ckpt\nhash: abc123\n"
        )

        result = loader.download(node)

        assert isinstance(result, DownloadTag)
        assert result.value == {
            "url": "https://example.com/model.ckpt",
            "hash": "abc123",
        }

    def test_loader_download_rejects_invalid_node_when_disabled(self):
        """Test disabled downloads reject non-scalar and non-mapping nodes."""
        loader = ConfigLoader("", download=False)
        node = yaml.compose("!download [a, b]")

        with pytest.raises(ConfigIncludeError, match="expects a string URL or dict"):
            loader.download(node)

    def test_loader_download_rejects_invalid_node_when_enabled(self):
        """Test enabled downloads reject non-scalar and non-mapping nodes."""
        loader = ConfigLoader("", download=True)
        node = yaml.compose("!download [a, b]")

        with pytest.raises(ConfigIncludeError, match="expects a string URL or dict"):
            loader.download(node)

    def test_loader_include_applies_collection_overrides_removals_and_strips_meta(
        self, monkeypatch
    ):
        """Test inline includes apply remaining overrides/removals and strip __meta__."""
        loader = ConfigLoader("", root_dir=os.getcwd())
        node = yaml.compose("!include child.yaml")

        def fake_recursive(*args, **kwargs):
            return (
                {META_KEY: {"version": "240719"}, "items": ["a"], "drop": 1},
                {"items+": "b"},
                ["drop"],
                {META_STRICT: "error", META_LIST_APPEND: "append"},
            )

        monkeypatch.setattr(
            "spine.config.loader.resolve_config_path",
            lambda *_args, **_kwargs: "child.yaml",
        )
        monkeypatch.setattr("spine.config.load._load_config_recursive", fake_recursive)

        result = loader.include(node)

        assert result == {"items": ["a", "b"]}
