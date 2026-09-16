"""Tests for framework-neutral direct dataset traversal."""

import sys

import numpy as np

import spine.io.manager as manager_module
from spine.driver import Driver
from spine.io import IOManager
from spine.io.dataset import larcv as larcv_dataset_module
from spine.io.dataset.base import BaseDataset
from spine.io.dataset.joint import JointDataset
from spine.io.read import HDF5Reader
from spine.io.write import HDF5Writer
from spine.utils.conditional import TORCH_AVAILABLE


def write_input(path):
    """Write a compact two-entry input file for direct-dataset tests."""
    writer = HDF5Writer(str(path))
    writer(
        {
            "index": np.asarray([0, 1]),
            "value": [
                np.asarray([1.0], dtype=np.float32),
                np.asarray([2.0], dtype=np.float32),
            ],
        },
        cfg={"io": {"writer": {"name": "hdf5"}}},
    )
    writer.finalize()
    writer.close()


def test_base_dataset_is_framework_neutral():
    """Map-style datasets should not inherit from an optional framework."""

    class ScalarDataset(BaseDataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"index": index}

    dataset = ScalarDataset()
    assert dataset[1] == {"index": 1}
    assert dataset.__getitems__([1, 0]) == [{"index": 1}, {"index": 0}]
    assert BaseDataset.__bases__ == (object,)


def test_direct_hdf5_dataset_round_trip_without_loader(tmp_path):
    """A direct dataset should read scalar products and write no Torch batches."""
    input_path = tmp_path / "input.h5"
    write_input(input_path)

    manager = IOManager(
        dataset={
            "name": "hdf5",
            "file_keys": str(input_path),
            "keys": ["value"],
            "keep_open": False,
        },
        iterations=-1,
    )

    assert manager.loader is None
    assert manager.iterations == 2
    first = manager.load(entry=0)
    assert first["index"] == 0
    np.testing.assert_array_equal(first["value"], np.asarray([1.0], dtype=np.float32))


def test_direct_larcv_dataset_parses_reader_products(monkeypatch):
    """Direct LArCV traversal should retain schema parsing without a loader."""

    class DummyParser:
        def __init__(self, dtype, raw_event):
            self.dtype = dtype
            self.raw_event = raw_event
            self.tree_keys = [raw_event]
            self.overlay_method = "cat"

        def __call__(self, data):
            return np.asarray(data[self.raw_event], dtype=self.dtype)

    class DummyReader:
        file_paths = ["input.root"]

        def __init__(self, tree_keys, **kwargs):
            assert tree_keys == ["raw"]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "index": index,
                "file_index": 0,
                "file_entry_index": index,
                "raw": [1.0, 2.0],
            }

    monkeypatch.setattr(larcv_dataset_module, "PARSER_DICT", {"dummy": DummyParser})
    monkeypatch.setattr(larcv_dataset_module, "LArCVReader", DummyReader)

    manager = IOManager(
        dataset={
            "name": "larcv",
            "file_keys": "input.root",
            "schema": {"value": {"parser": "dummy", "raw_event": "raw"}},
        }
    )
    data = manager.load(entry=0)

    assert data["index"] == 0
    assert np.issubdtype(data["value"].dtype, np.floating)
    np.testing.assert_array_equal(data["value"], np.asarray([1.0, 2.0]))


def test_direct_joint_dataset_cycles_secondary_entries(monkeypatch, tmp_path):
    """Direct joint traversal should write one event per primary entry."""

    class DummyReader:
        file_paths = ["input.root"]

    class ScalarDataset(BaseDataset):
        data_keys = ("index", "value")
        overlay_methods = {"index": "cat", "value": "sum"}

        def __init__(self, values):
            super().__init__()
            self.values = values
            self.reader = DummyReader()

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return {"index": index, "value": self.values[index]}

    dataset = JointDataset(
        primary=ScalarDataset([1.0, 2.0, 3.0]),
        secondary=ScalarDataset([10.0, 20.0]),
    )
    monkeypatch.setattr(
        manager_module, "dataset_factory", lambda *args, **kwargs: dataset
    )
    output_path = tmp_path / "joint.h5"
    manager = IOManager(
        dataset={"name": "joint"},
        writer={
            "name": "hdf5",
            "file_name": str(output_path),
            "overwrite": True,
        },
        iterations=-1,
    )

    for entry in range(manager.iterations):
        data = manager.load(entry=entry)
        manager.write(data, {})
    manager.close()

    reader = HDF5Reader(str(output_path), keep_open=False)
    assert len(reader) == 3
    assert [reader[index]["value"] for index in range(3)] == [11.0, 22.0, 13.0]


def test_direct_joint_dataset_stops_at_primary_length():
    """A longer secondary source should not extend direct joint traversal."""

    class ScalarDataset(BaseDataset):
        data_keys = ("index",)
        overlay_methods = {"index": "cat"}

        def __init__(self, size):
            super().__init__()
            self.size = size
            self.reader = object()

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            return {"index": index}

    dataset = JointDataset(
        primary=ScalarDataset(2),
        secondary=ScalarDataset(5),
    )

    assert len(dataset) == 2
    assert [dataset.sequential_pair_index(index) for index in range(len(dataset))] == [
        (0, 0),
        (1, 1),
    ]


def test_driver_converts_direct_dataset_without_loader(tmp_path):
    """The complete driver should copy event products without Torch collation."""
    input_path = tmp_path / "input.h5"
    output_path = tmp_path / "output.h5"
    write_input(input_path)

    driver = Driver(
        {
            "base": {
                "iterations": -1,
                "world_size": 0,
                "log_dir": str(tmp_path / "logs"),
                "overwrite_log": True,
            },
            "io": {
                "dataset": {
                    "name": "hdf5",
                    "file_keys": str(input_path),
                    "keys": ["value"],
                    "keep_open": False,
                },
                "writer": {
                    "name": "hdf5",
                    "file_name": str(output_path),
                    "overwrite": True,
                },
            },
        }
    )
    driver.run()

    reader = HDF5Reader(str(output_path), keep_open=False)
    assert len(reader) == 2
    np.testing.assert_array_equal(
        reader[1]["value"], np.asarray([2.0], dtype=np.float32)
    )
    if not TORCH_AVAILABLE:
        assert "torch" not in sys.modules
