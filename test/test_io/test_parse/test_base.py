"""Tests for generic parser base utilities."""

import pytest

from spine.io.parse.base import ParserBase


class DummyParser(ParserBase):
    """Minimal parser used to test ParserBase."""

    name = "dummy"
    returns = "scalar"
    overlay = "first"

    def __call__(self, trees):
        data = self.get_input_data(trees)
        return len(data)


def test_parser_base_collects_tree_keys():
    """ParserBase should collect unique tree keys from scalar and list inputs."""
    parser = DummyParser(
        dtype="float32",
        one_event="tree_a",
        many_event=["tree_b", "tree_a"],
    )

    assert parser.ftype == "float32"
    assert parser.itype == "int32"
    assert parser.data_map == {
        "one_event": "tree_a",
        "many_event": ["tree_b", "tree_a"],
    }
    assert parser.tree_keys == ["tree_a", "tree_b"]


def test_parser_base_rejects_unexpected_argument():
    """ParserBase should reject arguments without the `_event` suffix."""
    with pytest.raises(TypeError, match="unexpected argument"):
        DummyParser(dtype="float32", bad_key="tree_a")


def test_parser_base_get_input_data():
    """ParserBase should map trees into parser inputs."""
    parser = DummyParser(
        dtype="float32",
        one_event="tree_a",
        many_event=["tree_b", "tree_c"],
    )

    data = parser.get_input_data({"tree_a": 1, "tree_b": 2, "tree_c": 3})
    assert data == {"one_event": 1, "many_event": [2, 3]}


def test_parser_base_missing_tree_errors():
    """ParserBase should fail clearly when a required tree is absent."""
    parser = DummyParser(dtype="float32", one_event="tree_a")

    with pytest.raises(ValueError, match="Must provide tree_a"):
        parser.get_input_data({})


def test_parser_base_missing_tree_errors_for_lists():
    """ParserBase should also fail clearly for missing list-valued tree inputs."""
    parser = DummyParser(dtype="float32", many_event=["tree_a", "tree_b"])

    with pytest.raises(ValueError, match="Must provide tree_b"):
        parser.get_input_data({"tree_a": 1})
