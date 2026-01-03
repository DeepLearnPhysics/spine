"""Tests for the config loader functionality."""

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

override:
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

override:
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

override:
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
        """Test that dot notation creates final key but not missing parents."""
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
base:
  world_size: 1

new:
  nested:
    deeply:
      nested: {}

override:
  new.nested.deeply.nested.value: 123
  nonexistent.path.value: 456  # Should be skipped
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["base"]["world_size"] == 1
        # This works because new.nested.deeply.nested exists
        assert cfg["new"]["nested"]["deeply"]["nested"]["value"] == 123
        # This should not create nonexistent
        assert "nonexistent" not in cfg

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

    def test_relative_include_in_subdirectory(self, tmp_path):
        """Test that includes resolve relative to the file containing them."""
        # Create a subdirectory
        subdir = tmp_path / "configs"
        subdir.mkdir()

        # Create base config in subdirectory
        base_config = subdir / "base.yaml"
        base_config.write_text(
            """
base:
  world_size: 1
  seed: 0
"""
        )

        # Create child config in same subdirectory that includes base
        child_config = subdir / "child.yaml"
        child_config.write_text(
            """
include: base.yaml

base:
  iterations: 100
"""
        )

        # Load the child config (should find base.yaml relative to child.yaml)
        cfg = load_config(str(child_config))

        # Check that base values are loaded
        assert cfg["base"]["world_size"] == 1
        assert cfg["base"]["seed"] == 0
        # Check that child values override/extend
        assert cfg["base"]["iterations"] == 100

    def test_relative_inline_include_in_subdirectory(self, tmp_path):
        """Test that !include tags resolve relative to the file containing them."""
        # Create a subdirectory
        subdir = tmp_path / "configs"
        subdir.mkdir()

        # Create network config in subdirectory
        network_config = subdir / "network.yaml"
        network_config.write_text(
            """
depth: 5
filters: 32
"""
        )

        # Create main config in same subdirectory with inline include
        main_config = subdir / "main.yaml"
        main_config.write_text(
            """
model:
  name: full_chain
  uresnet: !include network.yaml
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["model"]["name"] == "full_chain"
        assert cfg["model"]["uresnet"]["depth"] == 5
        assert cfg["model"]["uresnet"]["filters"] == 32

    def test_absolute_path_include(self, tmp_path):
        """Test that absolute paths work for includes."""
        # Create base config
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
base:
  world_size: 1
"""
        )

        # Create main config with absolute path include
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            f"""
include: {str(base_config)}

base:
  iterations: 100
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["base"]["world_size"] == 1
        assert cfg["base"]["iterations"] == 100

    def test_included_file_with_removals(self, tmp_path):
        """Test that override/remove directives in included files are respected."""
        # Create a base config with full configuration
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
  - parser_two
  - parser_three
  - parser_four

model:
  name: full_chain
  depth: 5
"""
        )

        # Create a modifier config that removes some parsers
        modifier_config = tmp_path / "modifier.yaml"
        modifier_config.write_text(
            """
override:
  parsers: [parser_one, parser_three]
  model.depth: 3
"""
        )

        # Create main config that includes both
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include:
  - base.yaml
  - modifier.yaml
"""
        )

        cfg = load_config(str(main_config))

        # The modifier's override should be applied after merging
        assert cfg["parsers"] == ["parser_one", "parser_three"]
        assert cfg["model"]["name"] == "full_chain"
        assert cfg["model"]["depth"] == 3

    def test_included_file_with_null_removals(self, tmp_path):
        """Test that null values in override blocks from included files delete keys."""
        # Create a base config
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  reader:
    batch_size: 4
    shuffle: true
    num_workers: 8
  writer:
    output_dir: /tmp
"""
        )

        # Create a modifier that removes some keys
        modifier_config = tmp_path / "modifier.yaml"
        modifier_config.write_text(
            """
override:
  io.reader.shuffle: null
  io.writer: null
"""
        )

        # Create main config that includes both
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include:
  - base.yaml
  - modifier.yaml
"""
        )

        cfg = load_config(str(main_config))

        # shuffle should be deleted
        assert "shuffle" not in cfg["io"]["reader"]
        assert cfg["io"]["reader"]["batch_size"] == 4
        assert cfg["io"]["reader"]["num_workers"] == 8

        # writer should be completely deleted
        assert "writer" not in cfg["io"]

    def test_list_append_operation(self, tmp_path):
        """Test appending to lists using + operator."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
  - parser_two
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  parsers+: [parser_three, parser_four]
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["parsers"] == [
            "parser_one",
            "parser_two",
            "parser_three",
            "parser_four",
        ]

    def test_list_append_single_value(self, tmp_path):
        """Test appending a single value to a list."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  parsers+: parser_two
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["parsers"] == ["parser_one", "parser_two"]

    def test_list_remove_operation(self, tmp_path):
        """Test removing from lists using - operator."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
  - parser_two
  - parser_three
  - parser_four
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  parsers-: [parser_two, parser_four]
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["parsers"] == ["parser_one", "parser_three"]

    def test_list_remove_single_value(self, tmp_path):
        """Test removing a single value from a list."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
  - parser_two
  - parser_three
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  parsers-: parser_two
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["parsers"] == ["parser_one", "parser_three"]

    def test_list_operations_combined(self, tmp_path):
        """Test combining append and remove operations on the same list."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
  - parser_two
  - parser_three
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  parsers-: parser_two
  parsers+: [parser_four, parser_five]
"""
        )

        cfg = load_config(str(main_config))

        # Remove is processed first in dict iteration, then append
        # But both are applied after merge, order depends on dict ordering
        assert "parser_two" not in cfg["parsers"]
        assert "parser_four" in cfg["parsers"]
        assert "parser_five" in cfg["parsers"]
        assert "parser_one" in cfg["parsers"]
        assert "parser_three" in cfg["parsers"]

    def test_list_operations_nested_path(self, tmp_path):
        """Test list operations on nested paths."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  writer:
    file_keys:
      - key1
      - key2
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  io.writer.file_keys+: [key3, key4]
  io.writer.file_keys-: key1
"""
        )

        cfg = load_config(str(main_config))

        assert "key1" not in cfg["io"]["writer"]["file_keys"]
        assert "key2" in cfg["io"]["writer"]["file_keys"]
        assert "key3" in cfg["io"]["writer"]["file_keys"]
        assert "key4" in cfg["io"]["writer"]["file_keys"]

    def test_list_append_to_nonexistent_creates_list(self, tmp_path):
        """Test that appending to a non-existent key creates a new list."""
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
base:
  value: 1

override:
  parsers+: [parser_one, parser_two]
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["parsers"] == ["parser_one", "parser_two"]

    def test_list_operation_on_non_list_raises_error(self, tmp_path):
        """Test that list operations on non-list values raise errors."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
value: 42
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  value+: 10
"""
        )

        with pytest.raises(ValueError, match="not a list"):
            load_config(str(main_config))

    def test_list_remove_from_nonexistent_raises_error(self, tmp_path):
        """Test that removing from non-existent list raises error."""
        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
base:
  value: 1

override:
  parsers-: parser_one
"""
        )

        with pytest.raises(ValueError, match="does not exist"):
            load_config(str(main_config))

    def test_list_operations_in_included_file(self, tmp_path):
        """Test that list operations in included files work correctly."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
  - parser_two
  - parser_three
"""
        )

        modifier_config = tmp_path / "modifier.yaml"
        modifier_config.write_text(
            """
override:
  parsers-: parser_two
  parsers+: parser_four
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include:
  - base.yaml
  - modifier.yaml
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["parsers"] == ["parser_one", "parser_three", "parser_four"]

    def test_delete_nonexistent_key_with_null(self, tmp_path):
        """Test that deleting a non-existent key with null raises an error."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  reader:
    batch_size: 4
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  io.wrtier: null  # Typo: should be 'writer'
"""
        )

        with pytest.raises(KeyError) as exc_info:
            load_config(str(main_config))

        assert "io.wrtier" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)
        assert "typos" in str(exc_info.value).lower()

    def test_delete_nonexistent_key_with_remove(self, tmp_path):
        """Test that deleting a non-existent key with remove directive raises an error."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  reader:
    batch_size: 4
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

remove: io.reader.batchsize  # Typo: should be 'batch_size'
"""
        )

        with pytest.raises(KeyError) as exc_info:
            load_config(str(main_config))

        assert "io.reader.batchsize" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)

    def test_delete_nonexistent_nested_path(self, tmp_path):
        """Test that deleting with a non-existent parent path raises an error."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  reader:
    batch_size: 4
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  io.writter.output_dir: null  # Typo: 'writter' doesn't exist
"""
        )

        with pytest.raises(KeyError) as exc_info:
            load_config(str(main_config))

        assert "io.writter.output_dir" in str(exc_info.value)
        assert "io.writter" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)

    def test_dict_remove_operation(self, tmp_path):
        """Test removing keys from dictionaries using - operator."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
model:
  layers:
    layer1:
      depth: 5
    layer2:
      depth: 3
    layer3:
      depth: 7
    layer4:
      depth: 2
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  model.layers-: [layer2, layer4]
"""
        )

        cfg = load_config(str(main_config))

        assert "layer1" in cfg["model"]["layers"]
        assert "layer2" not in cfg["model"]["layers"]
        assert "layer3" in cfg["model"]["layers"]
        assert "layer4" not in cfg["model"]["layers"]

    def test_dict_remove_single_key(self, tmp_path):
        """Test removing a single key from a dictionary."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  writer:
    output_dir: /tmp
    format: hdf5
    compression: gzip
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  io.writer-: compression
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["io"]["writer"]["output_dir"] == "/tmp"
        assert cfg["io"]["writer"]["format"] == "hdf5"
        assert "compression" not in cfg["io"]["writer"]

    def test_dict_remove_nonexistent_key_is_safe(self, tmp_path):
        """Test that removing a non-existent key from dict doesn't raise error."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  writer:
    output_dir: /tmp
    format: hdf5
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  io.writer-: [nonexistent_key, format]
"""
        )

        cfg = load_config(str(main_config))

        # Should succeed, removing only format (nonexistent_key is safely ignored)
        assert cfg["io"]["writer"]["output_dir"] == "/tmp"
        assert "format" not in cfg["io"]["writer"]
        assert "nonexistent_key" not in cfg["io"]["writer"]

    def test_dict_append_raises_error(self, tmp_path):
        """Test that trying to append to a dict raises an error."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
model:
  layers:
    layer1:
      depth: 5
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  model.layers+: {layer2: {depth: 3}}
"""
        )

        with pytest.raises(ValueError, match="not supported for dicts"):
            load_config(str(main_config))

    def test_dict_operations_in_included_file(self, tmp_path):
        """Test that dict operations in included files work correctly."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  writer:
    key1: value1
    key2: value2
    key3: value3
    key4: value4
"""
        )

        modifier_config = tmp_path / "modifier.yaml"
        modifier_config.write_text(
            """
override:
  io.writer-: [key2, key4]
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include:
  - base.yaml
  - modifier.yaml
"""
        )

        cfg = load_config(str(main_config))

        assert cfg["io"]["writer"]["key1"] == "value1"
        assert "key2" not in cfg["io"]["writer"]
        assert cfg["io"]["writer"]["key3"] == "value3"
        assert "key4" not in cfg["io"]["writer"]

    def test_multiple_modifiers_with_list_operations(self, tmp_path):
        """Test that successive modifiers with list operations are accumulated."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
  - parser_two
  - parser_three
  - parser_four
  - parser_five
  - parser_six
"""
        )

        mod1_config = tmp_path / "mod1.yaml"
        mod1_config.write_text(
            """
override:
  parsers-: [parser_two, parser_four]
"""
        )

        mod2_config = tmp_path / "mod2.yaml"
        mod2_config.write_text(
            """
override:
  parsers-: [parser_five, parser_six]
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include:
  - base.yaml
  - mod1.yaml
  - mod2.yaml
"""
        )

        cfg = load_config(str(main_config))

        # All removals should be applied
        assert cfg["parsers"] == ["parser_one", "parser_three"]

    def test_multiple_modifiers_with_dict_operations(self, tmp_path):
        """Test that successive modifiers with dict operations are accumulated."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  writer:
    foo: 1
    bar: 2
    baz: 3
    boo: 4
    qux: 5
"""
        )

        mod1_config = tmp_path / "mod1.yaml"
        mod1_config.write_text(
            """
override:
  io.writer-: [foo, bar]
"""
        )

        mod2_config = tmp_path / "mod2.yaml"
        mod2_config.write_text(
            """
override:
  io.writer-: [baz, boo]
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include:
  - base.yaml
  - mod1.yaml
  - mod2.yaml
"""
        )

        cfg = load_config(str(main_config))

        # All keys should be removed
        assert "foo" not in cfg["io"]["writer"]
        assert "bar" not in cfg["io"]["writer"]
        assert "baz" not in cfg["io"]["writer"]
        assert "boo" not in cfg["io"]["writer"]
        assert cfg["io"]["writer"]["qux"] == 5

    def test_multiple_modifiers_with_list_appends(self, tmp_path):
        """Test that successive modifiers with list appends are accumulated."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
parsers:
  - parser_one
"""
        )

        mod1_config = tmp_path / "mod1.yaml"
        mod1_config.write_text(
            """
override:
  parsers+: [parser_two, parser_three]
"""
        )

        mod2_config = tmp_path / "mod2.yaml"
        mod2_config.write_text(
            """
override:
  parsers+: [parser_four, parser_five]
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include:
  - base.yaml
  - mod1.yaml
  - mod2.yaml
"""
        )

        cfg = load_config(str(main_config))

        # All appends should be applied
        assert cfg["parsers"] == [
            "parser_one",
            "parser_two",
            "parser_three",
            "parser_four",
            "parser_five",
        ]

    def test_override_only_if_parent_exists(self, tmp_path):
        """Test that overrides are only applied if parent path exists."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  reader:
    batch_size: 4
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  io.reader.batch_size: 8
  io.writer.output_dir: /tmp  # io.writer doesn't exist, should be skipped
  model.name: full_chain      # model doesn't exist, should be skipped
"""
        )

        cfg = load_config(str(main_config))

        # io.reader.batch_size should be overridden (parent exists)
        assert cfg["io"]["reader"]["batch_size"] == 8

        # io.writer and model should NOT be created
        assert "writer" not in cfg["io"]
        assert "model" not in cfg

    def test_override_nested_only_if_exists(self, tmp_path):
        """Test that deeply nested overrides are skipped if intermediate path missing."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  reader:
    batch_size: 4
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  io.reader.shuffle: true           # io.reader exists, should work
  io.loader.dataset.name: larcv     # io.loader doesn't exist, skip
"""
        )

        cfg = load_config(str(main_config))

        # io.reader.shuffle should be added (parent exists)
        assert cfg["io"]["reader"]["shuffle"] is True
        assert cfg["io"]["reader"]["batch_size"] == 4

        # io.loader should NOT be created
        assert "loader" not in cfg["io"]

    def test_override_creates_final_key_if_parent_exists(self, tmp_path):
        """Test that override creates the final key if its parent exists."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
io:
  reader:
    batch_size: 4
"""
        )

        main_config = tmp_path / "main.yaml"
        main_config.write_text(
            """
include: base.yaml

override:
  io.reader.new_key: new_value
"""
        )

        cfg = load_config(str(main_config))

        # new_key should be created because io.reader exists
        assert cfg["io"]["reader"]["new_key"] == "new_value"
        assert cfg["io"]["reader"]["batch_size"] == 4
