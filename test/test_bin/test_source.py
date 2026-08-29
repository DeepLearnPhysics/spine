"""Tests for command-line input-source parsing and routing."""

import pytest

from spine.bin import source as source_module


def test_parse_source_overrides_preserves_flat_inputs():
    """Unqualified direct paths and list files retain their existing shape."""
    assert source_module.parse_source_overrides(["a.root", "b.root"], None) == {
        None: {"file_keys": ["a.root", "b.root"]}
    }
    assert source_module.parse_source_overrides(None, "files.txt") == {
        None: {"file_list": "files.txt"}
    }
    assert source_module.parse_source_overrides(None, None) == {}


@pytest.mark.parametrize(
    ("source", "source_list", "message"),
    [
        (["raw.root"], ["hdf5=cache.txt"], "cannot be mixed"),
        (["raw.root"], ["files.txt"], "mutually exclusive"),
        (None, ["one.txt", "two.txt"], "exactly one"),
        (["=raw.root"], None, "Expected"),
        (["larcv="], None, "Expected"),
        (["larcv=raw.root"], ["larcv=raw.txt"], "both"),
        (None, ["larcv=one.txt", "larcv=two.txt"], "multiple"),
    ],
)
def test_parse_source_overrides_rejects_ambiguous_inputs(
    source,
    source_list,
    message,
):
    """Malformed, mixed and conflicting source selectors should fail early."""
    with pytest.raises(ValueError, match=message):
        source_module.parse_source_overrides(source, source_list)


@pytest.mark.parametrize(
    ("io_cfg", "source", "message", "error"),
    [
        (
            {"loader": {"dataset": {"name": "mixed", "larcv": {}, "hdf5": {}}}},
            ["raw.root"],
            "requires target-qualified",
            ValueError,
        ),
        (
            {"loader": {"dataset": {"name": "hdf5"}}},
            ["hdf5=cache.h5"],
            "require an inline joint or mixed",
            ValueError,
        ),
        (
            {"loader": {"dataset": {"name": "mixed", "larcv": {}, "hdf5": {}}}},
            ["primary=raw.root"],
            "Unknown source target",
            ValueError,
        ),
        (
            {"loader": {"dataset": {"name": "joint", "primary": {}}}},
            ["secondary=pileup.root"],
            "no `secondary` source block",
            KeyError,
        ),
        (
            {
                "loader": {
                    "dataset": {
                        "name": "joint",
                        "primary": "primary.yaml",
                        "secondary": {},
                    }
                }
            },
            ["primary=raw.root"],
            "inline `primary` source block",
            TypeError,
        ),
    ],
)
def test_apply_source_overrides_rejects_incompatible_configs(
    io_cfg,
    source,
    message,
    error,
):
    """Qualified selectors should match an inline composite source block."""
    with pytest.raises(error, match=message):
        source_module.apply_source_overrides(io_cfg, source, None)


@pytest.mark.parametrize(
    ("io_cfg", "message"),
    [
        ({"reader": "reader.yaml"}, "inline `io.reader`"),
        ({"loader": "loader.yaml"}, "`io.loader` block must be a mapping"),
        (
            {"loader": {"dataset": "dataset.yaml"}},
            "inline `io.loader.dataset`",
        ),
    ],
)
def test_input_config_rejects_external_blocks(io_cfg, message):
    """CLI input overrides require mutable inline configuration blocks."""
    with pytest.raises(TypeError, match=message):
        source_module.get_input_config(io_cfg)
