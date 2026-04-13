"""Tests for the SPINE config inspection command line tool."""

from io import StringIO

import yaml

from spine.bin.config import cli, diff_configs, dump_config, resolved_config_yaml


def test_resolved_config_yaml_expands_includes_and_overrides(tmp_path):
    """Test that dump output contains the fully resolved config."""
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        """
base:
  iterations: 100
io:
  reader:
    batch_size: 4
    shuffle: false
""",
        encoding="utf-8",
    )

    main_config = tmp_path / "main.yaml"
    main_config.write_text(
        """
include: base.yaml

override:
  io.reader.batch_size: 16
""",
        encoding="utf-8",
    )

    rendered = resolved_config_yaml(str(main_config))
    cfg = yaml.safe_load(rendered)

    assert cfg["base"]["iterations"] == 100
    assert cfg["io"]["reader"]["batch_size"] == 16
    assert "include" not in cfg
    assert "override" not in cfg


def test_dump_config_writes_output_file(tmp_path):
    """Test dumping a resolved config to a file."""
    config = tmp_path / "config.yaml"
    config.write_text("base:\n  iterations: 100\n", encoding="utf-8")
    output = tmp_path / "resolved.yaml"

    dump_config(str(config), str(output))

    assert yaml.safe_load(output.read_text(encoding="utf-8")) == {
        "base": {"iterations": 100}
    }


def test_diff_configs_compares_resolved_yaml(tmp_path):
    """Test diffing resolved configs rather than raw source files."""
    base_config = tmp_path / "base.yaml"
    base_config.write_text("base:\n  iterations: 100\n", encoding="utf-8")

    config_a = tmp_path / "a.yaml"
    config_a.write_text("include: base.yaml\n", encoding="utf-8")

    config_b = tmp_path / "b.yaml"
    config_b.write_text(
        """
include: base.yaml

base:
  iterations: 200
""",
        encoding="utf-8",
    )

    diff = diff_configs(str(config_a), str(config_b))

    assert "-  iterations: 100" in diff
    assert "+  iterations: 200" in diff


def test_cli_diff_exit_codes(tmp_path):
    """Test that diff returns shell-style status codes."""
    config_a = tmp_path / "a.yaml"
    config_a.write_text("base:\n  iterations: 100\n", encoding="utf-8")

    config_b = tmp_path / "b.yaml"
    config_b.write_text("base:\n  iterations: 200\n", encoding="utf-8")

    same_stream = StringIO()
    same_status = cli(["diff", str(config_a), str(config_a)], stream=same_stream)
    assert same_status == 0
    assert same_stream.getvalue() == ""

    diff_stream = StringIO()
    diff_status = cli(["diff", str(config_a), str(config_b)], stream=diff_stream)
    assert diff_status == 1
    assert "-  iterations: 100" in diff_stream.getvalue()
    assert "+  iterations: 200" in diff_stream.getvalue()
