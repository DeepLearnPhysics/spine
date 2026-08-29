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


def test_apply_validation_source_overrides_for_mixed_dataset():
    """Validation flags should populate the named composite-source schema."""
    io_cfg = {
        "loader": {
            "dataset": {
                "name": "mixed",
                "larcv": {"file_keys": ["train.root"]},
                "hdf5": {"file_keys": ["train.h5"]},
            }
        }
    }
    validation = {
        "file_keys": ["stale.root"],
        "fraction": 0.5,
    }

    source_module.apply_validation_source_overrides(
        validation,
        io_cfg,
        ["hdf5=/cache/val.h5"],
        ["larcv=validation.txt"],
    )

    assert validation == {
        "sources": {
            "larcv": {"file_list": "validation.txt"},
            "hdf5": {"file_keys": ["/cache/val.h5"]},
        },
        "fraction": 0.5,
    }

    untouched = {}
    source_module.apply_validation_source_overrides(untouched, {}, None, None)
    assert untouched == {}


def test_apply_validation_source_overrides_preserves_unmentioned_target():
    """A qualified override should retain other configured validation sources."""
    io_cfg = {
        "loader": {
            "dataset": {
                "name": "joint",
                "primary": {},
                "secondary": {},
            }
        }
    }
    validation = {
        "sources": {
            "primary": {"file_list": "primary.txt"},
            "secondary": {"file_list": "secondary.txt"},
        }
    }

    source_module.apply_validation_source_overrides(
        validation,
        io_cfg,
        ["primary=new.root"],
        None,
    )

    assert validation["sources"] == {
        "primary": {"file_keys": ["new.root"]},
        "secondary": {"file_list": "secondary.txt"},
    }


@pytest.mark.parametrize(
    ("io_cfg", "validation", "source", "message", "error"),
    [
        (
            {"loader": {"dataset": {"name": "mixed"}}},
            {},
            ["validation.root"],
            "requires target-qualified",
            ValueError,
        ),
        (
            {"loader": {"dataset": {"name": "hdf5"}}},
            {},
            ["hdf5=validation.h5"],
            "require an inline joint or mixed",
            ValueError,
        ),
        (
            {"loader": {"dataset": {"name": "mixed"}}},
            {},
            ["primary=validation.root"],
            "Unknown validation source target",
            ValueError,
        ),
        (
            {"loader": {"dataset": {"name": "mixed"}}},
            {},
            ["larcv=validation.root"],
            "must provide exactly",
            ValueError,
        ),
        (
            {"loader": {"dataset": {"name": "mixed"}}},
            {"sources": "validation.yaml"},
            ["larcv=validation.root"],
            "validation.sources.*inline mapping",
            TypeError,
        ),
    ],
)
def test_apply_validation_source_overrides_rejects_incompatible_configs(
    io_cfg,
    validation,
    source,
    message,
    error,
):
    """Validation selectors must match a complete dataset topology."""
    with pytest.raises(error, match=message):
        source_module.apply_validation_source_overrides(
            validation,
            io_cfg,
            source,
            None,
        )
