"""Tests for the config loader functionality."""

import os
import tempfile

import pytest

from spine.utils.config import load_config


class TestConfigLoader:
    """Test suite for the advanced YAML config loader."""

    def test_basic_load(self, tmp_path):
        """Test basic YAML loading without any special features."""
        config_file = tmp_path / "basic.yaml"
        config_file.write_text(
            """
base:
  world_size: 1
  iterations: 100
io:
  reader:
    batch_size: 4
"""
        )

        cfg = load_config(str(config_file))

        assert cfg["base"]["world_size"] == 1
        assert cfg["base"]["iterations"] == 100
        assert cfg["io"]["reader"]["batch_size"] == 4

    def test_top_level_include(self, tmp_path):
        """Test including another YAML file at the top level."""
        # Create base config
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
base:
  world_size: 1
  iterations: 100
  seed: 0
io:
  reader:
    batch_size: 4
    shuffle: false
"""
        )

        # Create main config that includes base
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

base:
  iterations: 200
"""
        )

        cfg = load_config(str(main_config))

        # Base values should be loaded
        assert cfg["base"]["world_size"] == 1
        assert cfg["base"]["seed"] == 0
        assert cfg["io"]["reader"]["batch_size"] == 4
        assert cfg["io"]["reader"]["shuffle"] is False

        # Override should work
        assert cfg["base"]["iterations"] == 200

    def test_multiple_includes(self, tmp_path):
        """Test including multiple YAML files."""
        # Create base config
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
base:
  world_size: 1
  iterations: 100
"""
        )

        # Create geometry config
        geo_config = tmp_path / "geo.yaml"
        geo_config.write_text(
            """
geo:
  detector: icarus
  tag: v4
"""
        )

        # Create main config that includes both
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include:
  - base.yaml
  - geo.yaml

io:
  reader:
    batch_size: 8
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["base"]["world_size"] == 1
        assert cfg["geo"]["detector"] == "icarus"
        assert cfg["io"]["reader"]["batch_size"] == 8

    def test_inline_include(self, tmp_path):
        """Test including a file inline within a block using !include."""
        # Create a separate config for reader settings
        reader_config = tmp_path / "reader_config.yaml"
        reader_config.write_text(
            """
batch_size: 4
shuffle: true
num_workers: 8
"""
        )

        # Create main config with inline include
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
base:
  world_size: 1

io:
  reader: !include reader_config.yaml
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["base"]["world_size"] == 1
        assert cfg["io"]["reader"]["batch_size"] == 4
        assert cfg["io"]["reader"]["shuffle"] is True
        assert cfg["io"]["reader"]["num_workers"] == 8

    def test_dot_notation_override(self, tmp_path):
        """Test modifying specific parameters using dot notation."""
        # Create base config
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
base:
  world_size: 1
  iterations: 100
io:
  reader:
    batch_size: 4
    file_paths: default.root
  writer:
    output_dir: /tmp/output
"""
        )

        # Create main config with overrides
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

io.reader.file_paths: /data/new_file.root
io.reader.batch_size: 16
base.iterations: 500
"""
        )

        cfg = load_config(str(main_config))

        # Overridden values
        assert cfg["io"]["reader"]["file_paths"] == "/data/new_file.root"
        assert cfg["io"]["reader"]["batch_size"] == 16
        assert cfg["base"]["iterations"] == 500

        # Non-overridden values should remain
        assert cfg["base"]["world_size"] == 1
        assert cfg["io"]["writer"]["output_dir"] == "/tmp/output"

    def test_dot_notation_with_list(self, tmp_path):
        """Test that dot notation can set list values."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  reader:
    file_paths: []
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

io.reader.file_paths: [file1.root, file2.root, file3.root]
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["io"]["reader"]["file_paths"] == [
            "file1.root",
            "file2.root",
            "file3.root",
        ]

    def test_nested_includes(self, tmp_path):
        """Test that included files can themselves include other files."""
        # Create level 2 config
        level2_config = tmp_path / "level2.yaml"
        level2_config.write_text(
            """
level2:
  value: 42
"""
        )

        # Create level 1 config that includes level 2
        level1_config = tmp_path / "level1.yaml"
        level1_config.write_text(
            """
include: level2.yaml

level1:
  value: 10
"""
        )

        # Create main config that includes level 1
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: level1.yaml

level0:
  value: 1
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["level0"]["value"] == 1
        assert cfg["level1"]["value"] == 10
        assert cfg["level2"]["value"] == 42

    def test_complex_scenario(self, tmp_path):
        """Test a complex scenario combining all features."""
        # Create base config
        base_config = tmp_path / "icarus_base.yaml"
        base_config.write_text(
            """
base:
  world_size: 1
  iterations: -1
  seed: 0

geo:
  detector: icarus
  tag: icarus_v4

io:
  loader:
    batch_size: 4
    shuffle: false
    dataset:
      name: larcv
      file_keys: null
"""
        )

        # Create a network config
        network_config = tmp_path / "network.yaml"
        network_config.write_text(
            """
depth: 5
filters: 32
num_classes: 5
"""
        )

        # Create main config
        main_config = tmp_path / "icarus_full_chain.yaml"
        main_config.write_text(
            """
include: icarus_base.yaml

model:
  name: full_chain
  modules:
    uresnet: !include network.yaml

io.loader.batch_size: 8
io.loader.dataset.file_keys: [data, seg_label]
base.iterations: 1000
"""
        )

        cfg = load_config(str(main_config))

        # Check base values are loaded
        assert cfg["base"]["world_size"] == 1
        assert cfg["base"]["seed"] == 0
        assert cfg["geo"]["detector"] == "icarus"
        assert cfg["geo"]["tag"] == "icarus_v4"

        # Check inline include worked
        assert cfg["model"]["name"] == "full_chain"
        assert cfg["model"]["modules"]["uresnet"]["depth"] == 5
        assert cfg["model"]["modules"]["uresnet"]["filters"] == 32

        # Check overrides worked
        assert cfg["base"]["iterations"] == 1000
        assert cfg["io"]["loader"]["batch_size"] == 8
        assert cfg["io"]["loader"]["dataset"]["file_keys"] == ["data", "seg_label"]

        # Check non-overridden values remain
        assert cfg["io"]["loader"]["shuffle"] is False
        assert cfg["io"]["loader"]["dataset"]["name"] == "larcv"

    def test_dot_notation_creates_nested_dicts(self, tmp_path):
        """Test that dot notation creates nested dictionaries if they don't exist."""
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
base:
  world_size: 1

new.nested.deeply.nested.value: 123
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["base"]["world_size"] == 1
        assert cfg["new"]["nested"]["deeply"]["nested"]["value"] == 123

    def test_include_file_not_found(self, tmp_path):
        """Test that missing include files raise appropriate error."""
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: nonexistent.yaml

base:
  value: 1
"""
        )

        with pytest.raises(FileNotFoundError):
            load_config(str(main_config))
