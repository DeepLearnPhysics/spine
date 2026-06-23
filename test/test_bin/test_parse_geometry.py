"""Tests for geometry parser helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script(path):
    """Import a bin script as a test module."""
    script_path = Path(__file__).resolve().parents[2] / path
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_larsoft_extract_version_handles_nonterminal_version_token():
    module = load_script(Path("bin/geo/parse_larsoft_geometry.py"))
    dune_tag = "dunevd10kt_3view_30deg_v3_refactored_1x8x6ref"

    assert module.extract_version("sbndv2") == 2
    assert module.extract_version(dune_tag) == 3


def test_flow_extract_version_keeps_existing_decimal_tag_behavior():
    module = load_script(Path("bin/geo/parse_flow_geometry.py"))
    dune_tag = "dunevd10kt_3view_30deg_v3_refactored_1x8x6ref"

    assert module.extract_version("mr6-5") == 6.5
    assert module.extract_version(dune_tag) == 3
    assert module.extract_version("untagged", fallback=7) == 7
