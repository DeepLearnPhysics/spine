"""Tests for shared command-line dataset selection."""

import argparse

import pytest

from spine.bin.dataset import (
    DatasetSelection,
    add_dataset_arguments,
    apply_dataset_selection,
    apply_validation_dataset_selection,
)


def test_dataset_arguments_are_symmetric():
    """Main and validation options should populate equivalent selections."""
    parser = argparse.ArgumentParser()
    add_dataset_arguments(parser)
    add_dataset_arguments(parser, validation=True)

    args = parser.parse_args(
        [
            "--source",
            "train.root",
            "--entry-fraction-range",
            "0.0",
            "0.5",
            "--val-source-list",
            "validation.txt",
            "--val-run-event-list",
            "events.txt",
        ]
    )
    train = DatasetSelection.from_namespace(args)
    validation = DatasetSelection.from_namespace(args, validation=True)

    assert train.source == ["train.root"]
    assert train.entry_fraction_range == (0.0, 0.5)
    assert validation.source_list == ["validation.txt"]
    assert validation.run_event_list == "events.txt"
    assert train.configured
    assert DatasetSelection().configured is False


def test_apply_dataset_selection_routes_ordinary_and_mixed_inputs():
    """Ordinary and aligned mixed inputs should store filters at their root."""
    ordinary = {
        "reader": {
            "file_list": "old.txt",
            "n_entry": 5,
            "entry_list": [1],
        }
    }
    apply_dataset_selection(
        ordinary,
        DatasetSelection(
            source=["new.root"],
            run_event_list="events.txt",
        ),
    )
    assert ordinary["reader"] == {
        "file_keys": ["new.root"],
        "file_list": None,
        "run_event_list": "events.txt",
    }
    unchanged = {"reader": dict(ordinary["reader"])}
    apply_dataset_selection(unchanged, DatasetSelection())
    assert unchanged["reader"] == ordinary["reader"]

    mixed = {
        "loader": {
            "dataset": {
                "name": "mixed",
                "larcv": {"file_keys": "raw.root"},
                "hdf5": {"file_keys": "cache.h5"},
            }
        }
    }
    apply_dataset_selection(
        mixed,
        DatasetSelection(entry_fraction_range=(0.25, 0.75)),
    )
    assert mixed["loader"]["dataset"]["entry_fraction_range"] == (0.25, 0.75)


def test_apply_dataset_selection_routes_joint_filters_to_primary():
    """Joint traversal filters should not restrict the secondary overlay pool."""
    io = {
        "loader": {
            "dataset": {
                "name": "joint",
                "base": {"entry_list": [1, 2]},
                "primary": {"file_keys": "primary.root", "n_skip": 2},
                "secondary": {"file_keys": "secondary.root", "n_skip": 3},
            }
        }
    }
    apply_dataset_selection(io, DatasetSelection(n_entry=10))

    dataset = io["loader"]["dataset"]
    assert dataset["primary"]["n_entry"] == 10
    assert "n_skip" not in dataset["primary"]
    assert dataset["primary"]["entry_list"] is None
    assert dataset["base"]["entry_list"] == [1, 2]
    assert dataset["secondary"]["n_skip"] == 3

    dataset["primary"] = "primary.yaml"
    with pytest.raises(TypeError, match="inline joint `primary`"):
        apply_dataset_selection(io, DatasetSelection(n_entry=2))


def test_validation_selection_preserves_or_replaces_entry_filters():
    """Source-only changes preserve filters; explicit filters replace the mode."""
    io = {"loader": {"dataset": {"file_keys": "train.root"}}}
    validation = {"file_keys": "old.root", "n_entry": 5}

    apply_validation_dataset_selection(
        validation,
        io,
        DatasetSelection(source=["new.root"]),
    )
    assert validation == {"file_keys": ["new.root"], "n_entry": 5}

    apply_validation_dataset_selection(
        validation,
        io,
        DatasetSelection(entry_fraction_range=(0.5, 1.0)),
    )
    assert validation == {
        "file_keys": ["new.root"],
        "entry_fraction_range": (0.5, 1.0),
    }
