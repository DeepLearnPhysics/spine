"""Tests for the structural HDF5 litification CLI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


def load_output_litify_module():
    """Import ``bin/output/output_litify.py`` as a test module."""
    script_path = (
        Path(__file__).resolve().parents[2] / "bin" / "output" / "output_litify.py"
    )
    spec = importlib.util.spec_from_file_location("output_litify", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_litify_cli_loads_simple_and_driver_configs(tmp_path):
    """Key selection should accept both concise and existing SPINE configs."""
    module = load_output_litify_module()
    simple = tmp_path / "simple.yaml"
    full = tmp_path / "full.yaml"
    simple.write_text(yaml.safe_dump({"keys": ["meta", "particles"]}))
    full.write_text(
        yaml.safe_dump({"io": {"writer": {"keys": ["run_info", "interactions"]}}})
    )

    assert module.resolve_keys(None, str(simple)) == ("meta", "particles")
    assert module.resolve_keys(None, str(full)) == (
        "run_info",
        "interactions",
    )
    assert module.resolve_keys(("particles",), None) == ("particles",)


def test_litify_cli_rejects_ambiguous_or_invalid_selection(tmp_path):
    """CLI and config selection should fail clearly when malformed."""
    module = load_output_litify_module()
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump({"keys": "particles"}))

    with pytest.raises(ValueError, match="either"):
        module.resolve_keys(("particles",), str(invalid))
    with pytest.raises(ValueError, match="string list"):
        module.load_keys(str(invalid))
