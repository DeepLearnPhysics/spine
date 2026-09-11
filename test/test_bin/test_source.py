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
            {"loader": {"dataset": {"name": "mixed", "primary": {}, "cache": {}}}},
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


def test_input_config_resolves_reader_and_rejects_missing_input():
    """Inline readers resolve directly and absent input fails clearly."""
    reader = {"name": "cache", "path": "input.spine-cache"}
    assert source_module.get_input_config({"reader": reader}) == (reader, False)
    with pytest.raises(KeyError, match="loader.*reader"):
        source_module.get_input_config({})


def test_apply_source_overrides_without_values_is_a_noop():
    """Omitted source options should not require an input configuration."""
    io_cfg = {}
    source_module.apply_source_overrides(io_cfg, None, None)
    assert io_cfg == {}


def test_apply_validation_source_overrides_for_mixed_dataset():
    """Validation flags should populate the named composite-source schema."""
    io_cfg = {
        "loader": {
            "dataset": {
                "name": "mixed",
                "primary": {"name": "larcv", "file_keys": ["train.root"]},
                "cache": {"name": "cache", "path": "train.spine-cache"},
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
        ["cache=/cache/validation.spine-cache"],
        ["primary=validation.txt"],
    )

    assert validation == {
        "sources": {
            "primary": {"file_list": "validation.txt"},
            "cache": {"path": "/cache/validation.spine-cache"},
        },
        "fraction": 0.5,
    }

    untouched = {}
    source_module.apply_validation_source_overrides(untouched, {}, None, None)
    assert untouched == {}


def test_apply_cache_source_overrides_use_repository_paths():
    """Cache inputs consume one repository path rather than HDF5 selectors."""
    io_cfg = {"loader": {"dataset": {"name": "cache", "path": "old"}}}
    source_module.apply_source_overrides(io_cfg, ["new.spine-cache"], None)
    assert io_cfg["loader"]["dataset"] == {
        "name": "cache",
        "path": "new.spine-cache",
    }

    with pytest.raises(ValueError, match="not valid for a cache repository"):
        source_module.apply_source_overrides(io_cfg, None, "repositories.txt")


def test_apply_canonical_mixed_cache_source_overrides():
    """Canonical mixed roles route cache repositories through ``path``."""
    io_cfg = {
        "loader": {
            "dataset": {
                "name": "mixed",
                "primary": {"name": "larcv"},
                "cache": {},
            }
        }
    }
    source_module.apply_source_overrides(
        io_cfg,
        ["primary=raw.root", "cache=train.spine-cache"],
        None,
    )
    dataset = io_cfg["loader"]["dataset"]
    assert dataset["primary"]["file_keys"] == ["raw.root"]
    assert dataset["cache"]["path"] == "train.spine-cache"
    with pytest.raises(ValueError, match="Unknown source target"):
        source_module.apply_source_overrides(io_cfg, ["other=value"], None)
    with pytest.raises(ValueError, match="Unknown source target 'larcv'"):
        source_module.apply_source_overrides(io_cfg, ["larcv=raw.root"], None)

    with pytest.raises(ValueError, match="exactly one --source path"):
        source_module.apply_source_overrides(
            {"loader": {"dataset": {"name": "cache"}}},
            ["one.spine-cache", "two.spine-cache"],
            None,
        )


def test_apply_cache_validation_source_overrides():
    """Validation cache selectors should use paths in flat and mixed modes."""
    io_cfg = {"loader": {"dataset": {"name": "cache", "path": "train"}}}
    validation = {}
    source_module.apply_validation_source_overrides(
        validation, io_cfg, ["validation.spine-cache"], None
    )
    assert validation == {"path": "validation.spine-cache"}

    mixed_io = {
        "loader": {
            "dataset": {
                "name": "mixed",
                "primary": {"name": "larcv"},
                "cache": {},
            }
        }
    }
    validation = {"sources": {"primary": {"file_list": "validation.txt"}}}
    source_module.apply_validation_source_overrides(
        validation,
        mixed_io,
        ["cache=validation.spine-cache"],
        None,
    )
    assert validation["sources"]["cache"] == {"path": "validation.spine-cache"}
    with pytest.raises(ValueError, match="Unknown validation source target"):
        source_module.apply_validation_source_overrides(
            {}, mixed_io, ["other=value"], None
        )

    invalid_io = {
        "loader": {
            "dataset": {
                "name": "mixed",
                "primary": "external.yaml",
                "cache": {"name": "cache"},
            }
        }
    }
    with pytest.raises(TypeError, match="inline `primary` block"):
        source_module.apply_validation_source_overrides(
            {"sources": {"cache": {"path": "validation.spine-cache"}}},
            invalid_io,
            ["primary=raw.root"],
            None,
        )


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
            {
                "loader": {
                    "dataset": {
                        "name": "mixed",
                        "primary": {"name": "larcv"},
                        "cache": {"name": "cache"},
                    }
                }
            },
            {},
            ["primary=validation.root"],
            "must provide exactly",
            ValueError,
        ),
        (
            {
                "loader": {
                    "dataset": {
                        "name": "mixed",
                        "primary": {"name": "larcv"},
                        "cache": {"name": "cache"},
                    }
                }
            },
            {"sources": "validation.yaml"},
            ["primary=validation.root"],
            "validation.sources.*inline mapping",
            TypeError,
        ),
        (
            {
                "loader": {
                    "dataset": {
                        "name": "mixed",
                        "primary": {"name": "larcv"},
                    }
                }
            },
            {},
            ["cache=validation.spine-cache"],
            "no `cache` source block",
            KeyError,
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
