"""Tests for shared LArCV script helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_larcv_utils():
    """Import ``bin/larcv/utils.py`` as a test module."""
    script_path = Path(__file__).resolve().parents[2] / "bin" / "larcv" / "utils.py"
    spec = importlib.util.spec_from_file_location("larcv_utils", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeKey:
    def __init__(self, name):
        self.name = name

    def GetName(self):
        return self.name


class FakeFile:
    def __init__(self):
        self.keys = [
            FakeKey("metadata"),
            FakeKey("sparse3d_reco_tree"),
        ]
        self.sparse3d_reco_tree = object()

    def GetListOfKeys(self):
        return self.keys

    def Get(self, name):
        return object()


def test_tree_lookup_uses_larcv_tree_names_and_typed_attributes():
    module = load_larcv_utils()
    root_file = FakeFile()

    assert module.list_tree_keys(root_file) == ["sparse3d_reco_tree"]
    assert module.get_tree_key(root_file) == "sparse3d_reco_tree"
    assert module.get_tree_key(root_file, "cluster3d_pcluster") == (
        "cluster3d_pcluster_tree"
    )
    assert module.get_branch_key("sparse3d_reco_tree") == "sparse3d_reco_branch"
    assert root_file.Get("sparse3d_reco_tree") is not root_file.sparse3d_reco_tree
    assert (
        module.get_tree(root_file, "sparse3d_reco_tree") is root_file.sparse3d_reco_tree
    )
