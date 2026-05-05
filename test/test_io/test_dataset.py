"""Test that the dataset classes work as intended."""

import importlib

import numpy as np
import pytest

from spine.data import IndexBatch, TensorBatch
from spine.io import dataset as dataset_module
from spine.io.collate import CollateAll
from spine.io.dataset import *
from spine.io.dataset import base as dataset_base_module
from spine.io.dataset import hdf5 as hdf5_dataset_module
from spine.io.dataset import larcv as larcv_dataset_module
from spine.io.dataset import mixed as mixed_dataset_module
from spine.io.write import HDF5Writer
from spine.utils.conditional import ROOT, ROOT_AVAILABLE, TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch is required for torch-backed IO datasets."
)


@pytest.mark.skipif(
    not ROOT_AVAILABLE, reason="ROOT is required for LArCV dataset tests."
)
def test_larcv_dataset(larcv_data):
    """Tests a torch dataset based on LArCV data.

    Most of the functions of this dataset are shared with the underlying
    :class:`LArCVReader` class which is tested elsewhere.
    """
    # Get the list of tree keys in the larcv file
    root_file = ROOT.TFile(larcv_data, "r")
    num_entries = None
    tree_keys = []
    for tree in root_file.GetListOfKeys():
        tree_keys.append(tree.GetName().split("_tree")[0])
        if num_entries is None:
            num_entries = getattr(root_file, tree.GetName()).GetEntries()

    root_file.Close()

    # Create a dummy schema based on the data keys
    schema = {}
    for key in tree_keys:
        datatype = key.split("_")[0]
        el = {}
        if datatype == "sparse3d":
            el["parser"] = datatype
            el["sparse_event"] = key
        elif datatype == "cluster3d":
            el["parser"] = datatype
            el["cluster_event"] = key
        elif datatype == "particle":
            el["parser"] = datatype
            el["particle_event"] = key
            el["pixel_coordinates"] = False
        else:
            raise ValueError("Unrecognized data product, cannot set up schema.")

        schema[key] = el

    # Initialize the dataset
    dataset = LArCVDataset(file_keys=larcv_data, schema=schema, dtype="float32")

    # Load the items in the dataset, check the keys
    for i, entry in enumerate(dataset):
        for key in tree_keys:
            assert key in entry
        assert "index" in entry
        assert entry["index"] == i

    # Check that the data keys are as expected
    for key in tree_keys:
        assert key in dataset.data_keys

    # Check that one can list the content of the dataset
    data_dict = dataset.list_data(larcv_data)
    data_keys = []
    for val in data_dict.values():
        data_keys += list(val)
    for key in tree_keys:
        assert key in data_keys


def test_hdf5_dataset(hdf5_data):
    """Tests the torch dataset wrapper around HDF5Reader."""
    dataset = HDF5Dataset(
        file_keys=hdf5_data,
        build_classes=False,
        keys=["run_info"],
    )

    assert len(dataset) > 0
    assert "index" in dataset.data_keys
    assert "run_info" in dataset.data_keys

    entry = dataset[0]
    assert entry["index"] == 0
    assert entry["file_index"] == 0
    assert entry["file_entry_index"] == 0
    assert "run_info" in entry


def test_hdf5_dataset_skip_keys(hdf5_data):
    """The HDF5 dataset should support dropping selected products."""
    dataset = HDF5Dataset(
        file_keys=hdf5_data,
        build_classes=False,
        skip_keys=["run_info"],
    )

    entry = dataset[0]
    assert "run_info" not in entry
    assert "index" in entry


def test_hdf5_dataset_rejects_keys_and_skip_keys(hdf5_data):
    """The HDF5 dataset should reject conflicting key selection options."""
    with pytest.raises(ValueError, match="Provide either `keys` or `skip_keys`"):
        HDF5Dataset(
            file_keys=hdf5_data,
            build_classes=False,
            keys=["run_info"],
            skip_keys=["run_info"],
        )


def test_hdf5_dataset_requires_dtype_with_schema(hdf5_data):
    """Schema-driven parsing requires an explicit dtype."""
    with pytest.raises(ValueError, match="explicit `dtype` is required"):
        HDF5Dataset(
            file_keys=hdf5_data,
            schema={
                "run_info": {
                    "parser": "feature_tensor",
                    "tensor_event": "run_info",
                }
            },
        )


def test_hdf5_dataset_uses_explicit_metadata(hdf5_data):
    """The HDF5 dataset should expose explicit collate metadata when provided."""
    dataset = HDF5Dataset(
        file_keys=hdf5_data,
        build_classes=False,
        keys=["run_info"],
        data_types={"run_info": "object"},
        overlay_methods={"run_info": "first"},
    )

    assert dataset.data_types["run_info"] == "object"
    assert dataset.overlay_methods["run_info"] == "first"


def test_hdf5_dataset_rejects_missing_torch(monkeypatch, hdf5_data):
    """The HDF5 dataset should fail clearly when torch support is disabled."""
    monkeypatch.setattr(hdf5_dataset_module, "TORCH_AVAILABLE", False)

    with pytest.raises(ImportError, match="PyTorch is required"):
        HDF5Dataset(file_keys=hdf5_data, build_classes=False)


def test_hdf5_dataset_with_augment(monkeypatch, hdf5_data):
    """The HDF5 dataset should apply augmentation when configured."""

    class DummyAugmenter:
        def __call__(self, result):
            result = dict(result)
            result["augmented"] = True
            return result

    monkeypatch.setattr(
        dataset_base_module, "AugmentManager", lambda **_: DummyAugmenter()
    )

    dataset = HDF5Dataset(
        file_keys=hdf5_data,
        build_classes=False,
        keys=["run_info"],
        augment={"name": "dummy"},
    )

    assert dataset[0]["augmented"] is True


def test_hdf5_dataset_with_schema_and_collate(tmp_path):
    """The HDF5 dataset should parse cached GrapPA inputs into parser products."""
    output = tmp_path / "grappa_cache.h5"
    writer = HDF5Writer(str(output))
    writer(
        {
            "index": np.asarray([0, 1]),
            "clusts": [
                [
                    np.asarray([0, 1], dtype=np.int64),
                    np.asarray([2], dtype=np.int64),
                ],
                [np.asarray([0], dtype=np.int64)],
            ],
            "node_features": [
                np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
                np.asarray([[7.0, 8.0]], dtype=np.float32),
            ],
            "edge_features": [
                np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                np.asarray([[0.5, 0.6]], dtype=np.float32),
            ],
        },
        cfg={"io": {"writer": {"name": "hdf5"}}},
    )
    writer.finalize()
    writer.close()

    schema = {
        "clusts": {
            "parser": "index_list",
            "index_event": "clusts",
            "count_event": "node_features",
        },
        "node_features": {
            "parser": "feature_tensor",
            "tensor_event": "node_features",
        },
        "edge_features": {
            "parser": "feature_tensor",
            "tensor_event": "edge_features",
        },
    }

    dataset = HDF5Dataset(file_keys=str(output), schema=schema, dtype="float32")
    entry = dataset[0]
    assert "clusts" in entry
    assert "node_features" in entry
    assert "edge_features" in entry
    assert dataset.data_keys == (
        "index",
        "file_index",
        "file_entry_index",
        "clusts",
        "node_features",
        "edge_features",
    )
    assert dataset.data_types["clusts"] == "tensor"
    assert dataset.overlay_methods["clusts"] is None

    collate = CollateAll(dataset.data_types)
    batch = collate([dataset[0], dataset[1]])

    assert isinstance(batch["clusts"], IndexBatch)
    assert isinstance(batch["node_features"], TensorBatch)
    assert isinstance(batch["edge_features"], TensorBatch)
    assert batch["clusts"].batch_size == 2
    assert batch["clusts"].counts.tolist() == [2, 1]
    assert batch["clusts"].single_counts.tolist() == [2, 1, 1]
    assert batch["node_features"].counts.tolist() == [3, 1]
    assert batch["edge_features"].counts.tolist() == [2, 1]


def test_hdf5_dataset_schema_updates_explicit_raw_keys(tmp_path):
    """Schema inference should merge required raw HDF5 keys into explicit selection."""
    output = tmp_path / "schema_keys.h5"
    writer = HDF5Writer(str(output))
    writer(
        {
            "index": np.asarray([0]),
            "clusts": [[np.asarray([0], dtype=np.int64)]],
            "node_features": [np.asarray([[1.0]], dtype=np.float32)],
        },
        cfg={"io": {"writer": {"name": "hdf5"}}},
    )
    writer.finalize()
    writer.close()

    dataset = HDF5Dataset(
        file_keys=str(output),
        dtype="float32",
        keys=["index"],
        schema={
            "clusts": {
                "parser": "index_list",
                "index_event": "clusts",
                "count_event": "node_features",
            }
        },
    )

    assert dataset[0]["clusts"].global_shift == 1


def test_hdf5_dataset_schema_parser_failure(tmp_path):
    """Schema parsing failures should be logged and re-raised."""
    output = tmp_path / "schema_failure.h5"
    writer = HDF5Writer(str(output))
    writer(
        {
            "index": np.asarray([0]),
            "node_features": [np.asarray([[1.0]], dtype=np.float32)],
        },
        cfg={"io": {"writer": {"name": "hdf5"}}},
    )
    writer.finalize()
    writer.close()

    class BrokenParser:
        tree_keys = ["node_features"]
        returns = "tensor"
        overlay = None

        def __call__(self, _data):
            raise RuntimeError("boom")

    dataset = HDF5Dataset(file_keys=str(output), build_classes=False)
    dataset.parsers = {"broken": BrokenParser()}
    dataset.keys = {"node_features"}

    with pytest.raises(RuntimeError, match="boom"):
        dataset[0]


def test_mixed_dataset_merges_aligned_sources(monkeypatch):
    """The mixed dataset should merge aligned LArCV and HDF5 samples."""

    class DummyDataset:
        def __init__(self, samples, data_types, overlay_methods):
            self.samples = samples
            self._data_types = data_types
            self._overlay_methods = overlay_methods
            self.reader = object()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return dict(self.samples[idx])

        @property
        def data_types(self):
            return self._data_types

        @property
        def overlay_methods(self):
            return self._overlay_methods

    larcv_samples = [
        {
            "index": 0,
            "file_index": 0,
            "file_entry_index": 0,
            "data": "larcv-data",
            "coord_label": "coords",
        }
    ]
    hdf5_samples = [
        {
            "index": 0,
            "file_index": 0,
            "file_entry_index": 0,
            "source_file_index": 0,
            "source_file_entry_index": 0,
            "clusts": "cached-clusts",
            "node_features": "cached-node-features",
        }
    ]

    monkeypatch.setattr(
        mixed_dataset_module,
        "LArCVDataset",
        lambda **kwargs: DummyDataset(
            larcv_samples,
            {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
                "data": "tensor",
                "coord_label": "tensor",
            },
            {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
                "data": None,
                "coord_label": None,
            },
        ),
    )
    monkeypatch.setattr(
        mixed_dataset_module,
        "HDF5Dataset",
        lambda **kwargs: DummyDataset(
            hdf5_samples,
            {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
                "source_file_index": "scalar",
                "source_file_entry_index": "scalar",
                "clusts": "tensor",
                "node_features": "tensor",
            },
            {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
                "source_file_index": "cat",
                "source_file_entry_index": "cat",
                "clusts": None,
                "node_features": None,
            },
        ),
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
    )

    entry = dataset[0]
    assert entry["data"] == "larcv-data"
    assert entry["coord_label"] == "coords"
    assert entry["clusts"] == "cached-clusts"
    assert entry["node_features"] == "cached-node-features"
    assert dataset.data_types["clusts"] == "tensor"
    assert dataset.overlay_methods["clusts"] is None
    assert set(dataset.data_keys) == {
        "index",
        "file_index",
        "file_entry_index",
        "source_file_index",
        "source_file_entry_index",
        "data",
        "coord_label",
        "clusts",
        "node_features",
    }


def test_mixed_dataset_rejects_alignment_mismatch(monkeypatch):
    """The mixed dataset should fail clearly when sources do not align."""

    class DummyDataset:
        def __init__(self, samples):
            self.samples = samples
            self.reader = object()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return dict(self.samples[idx])

        @property
        def data_types(self):
            return {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
            }

        @property
        def overlay_methods(self):
            return {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
            }

    monkeypatch.setattr(
        mixed_dataset_module,
        "LArCVDataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "file_index": 0, "file_entry_index": 0}]
        ),
    )
    monkeypatch.setattr(
        mixed_dataset_module,
        "HDF5Dataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "file_index": 0, "file_entry_index": 1}]
        ),
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
    )

    with pytest.raises(ValueError, match="alignment failed"):
        dataset[0]


def test_mixed_dataset_prefers_source_provenance(monkeypatch):
    """The mixed dataset should align on persisted source provenance by default."""

    class DummyDataset:
        def __init__(self, samples):
            self.samples = samples
            self.reader = object()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return dict(self.samples[idx])

        @property
        def data_types(self):
            return {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
            }

        @property
        def overlay_methods(self):
            return {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
            }

    monkeypatch.setattr(
        mixed_dataset_module,
        "LArCVDataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "file_index": 0, "file_entry_index": 5}]
        ),
    )
    monkeypatch.setattr(
        mixed_dataset_module,
        "HDF5Dataset",
        lambda **kwargs: DummyDataset(
            [
                {
                    "index": 0,
                    "file_index": 99,
                    "file_entry_index": 42,
                    "source_file_index": 0,
                    "source_file_entry_index": 5,
                }
            ]
        ),
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
    )

    entry = dataset[0]
    assert entry["file_index"] == 0
    assert entry["file_entry_index"] == 5


def test_mixed_dataset_len_mismatch_raises(monkeypatch):
    """The mixed dataset should reject sources with different lengths."""

    class DummyDataset:
        def __init__(self, size):
            self.size = size
            self.reader = object()

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"index": idx, "file_index": 0, "file_entry_index": idx}

        @property
        def data_types(self):
            return {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
            }

        @property
        def overlay_methods(self):
            return {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
            }

    monkeypatch.setattr(
        mixed_dataset_module, "LArCVDataset", lambda **kwargs: DummyDataset(2)
    )
    monkeypatch.setattr(
        mixed_dataset_module, "HDF5Dataset", lambda **kwargs: DummyDataset(3)
    )

    with pytest.raises(ValueError, match="same number of entries"):
        MixedDataset(
            larcv={"file_keys": "dummy.root", "schema": {}},
            hdf5={"file_keys": "dummy.h5"},
            dtype="float32",
        )


def test_mixed_dataset_len_delegates_to_primary(monkeypatch):
    """The mixed dataset length should come from the primary source."""

    class DummyDataset:
        def __init__(self, size):
            self.size = size
            self.reader = object()

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"index": idx, "file_index": 0, "file_entry_index": idx}

        @property
        def data_types(self):
            return {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
            }

        @property
        def overlay_methods(self):
            return {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
            }

    monkeypatch.setattr(
        mixed_dataset_module, "LArCVDataset", lambda **kwargs: DummyDataset(4)
    )
    monkeypatch.setattr(
        mixed_dataset_module, "HDF5Dataset", lambda **kwargs: DummyDataset(4)
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
    )
    assert len(dataset) == 4


def test_mixed_dataset_respects_explicit_hdf5_align_keys(monkeypatch):
    """Explicit HDF5 alignment mappings should override automatic source_* lookup."""

    class DummyDataset:
        def __init__(self, samples):
            self.samples = samples
            self.reader = object()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return dict(self.samples[idx])

        @property
        def data_types(self):
            return {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
            }

        @property
        def overlay_methods(self):
            return {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
            }

    monkeypatch.setattr(
        mixed_dataset_module,
        "LArCVDataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "file_index": 0, "file_entry_index": 5}]
        ),
    )
    monkeypatch.setattr(
        mixed_dataset_module,
        "HDF5Dataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "cache_file_id": 0, "cache_entry_id": 5}]
        ),
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
        hdf5_align_keys={
            "file_index": "cache_file_id",
            "file_entry_index": "cache_entry_id",
        },
    )

    entry = dataset[0]
    assert entry["file_entry_index"] == 5


def test_mixed_dataset_key_collision_can_be_overwritten(monkeypatch):
    """Explicit overwrite should let cached values replace LArCV values."""

    class DummyDataset:
        def __init__(self, samples, data_types=None, overlay_methods=None):
            self.samples = samples
            self.reader = object()
            self._data_types = data_types or {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
            }
            self._overlay_methods = overlay_methods or {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
            }

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return dict(self.samples[idx])

        @property
        def data_types(self):
            return self._data_types

        @property
        def overlay_methods(self):
            return self._overlay_methods

    monkeypatch.setattr(
        mixed_dataset_module,
        "LArCVDataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "file_index": 0, "file_entry_index": 0, "shared": "larcv"}],
            {"index": "scalar", "file_index": "scalar", "file_entry_index": "scalar"},
            {"index": "cat", "file_index": "cat", "file_entry_index": "cat"},
        ),
    )
    monkeypatch.setattr(
        mixed_dataset_module,
        "HDF5Dataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "file_index": 0, "file_entry_index": 0, "shared": "cache"}],
            {"index": "scalar", "file_index": "scalar", "file_entry_index": "scalar"},
            {"index": "cat", "file_index": "cat", "file_entry_index": "cat"},
        ),
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
        allow_overwrite=True,
    )

    assert dataset[0]["shared"] == "cache"


def test_mixed_dataset_key_collision_raises(monkeypatch):
    """Mixed dataset should reject colliding cache keys by default."""

    class DummyDataset:
        def __init__(self, samples):
            self.samples = samples
            self.reader = object()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return dict(self.samples[idx])

        @property
        def data_types(self):
            return {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
            }

        @property
        def overlay_methods(self):
            return {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
            }

    monkeypatch.setattr(
        mixed_dataset_module,
        "LArCVDataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "file_index": 0, "file_entry_index": 0, "shared": "larcv"}]
        ),
    )
    monkeypatch.setattr(
        mixed_dataset_module,
        "HDF5Dataset",
        lambda **kwargs: DummyDataset(
            [{"index": 0, "file_index": 0, "file_entry_index": 0, "shared": "cache"}]
        ),
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
    )

    with pytest.raises(ValueError, match="key collision"):
        dataset[0]


def test_mixed_dataset_data_type_collision_raises(monkeypatch):
    """Mixed dataset should reject incompatible collate type metadata."""

    class DummyDataset:
        def __init__(self, data_types, overlay_methods):
            self.reader = object()
            self._data_types = data_types
            self._overlay_methods = overlay_methods

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"index": 0, "file_index": 0, "file_entry_index": 0}

        @property
        def data_types(self):
            return self._data_types

        @property
        def overlay_methods(self):
            return self._overlay_methods

    monkeypatch.setattr(
        mixed_dataset_module,
        "LArCVDataset",
        lambda **kwargs: DummyDataset(
            {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
                "shared": "tensor",
            },
            {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
                "shared": None,
            },
        ),
    )
    monkeypatch.setattr(
        mixed_dataset_module,
        "HDF5Dataset",
        lambda **kwargs: DummyDataset(
            {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
                "shared": "object",
            },
            {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
                "shared": None,
            },
        ),
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
    )

    with pytest.raises(ValueError, match="data type collision"):
        _ = dataset.data_types


def test_mixed_dataset_overlay_collision_raises(monkeypatch):
    """Mixed dataset should reject incompatible overlay metadata."""

    class DummyDataset:
        def __init__(self, data_types, overlay_methods):
            self.reader = object()
            self._data_types = data_types
            self._overlay_methods = overlay_methods

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"index": 0, "file_index": 0, "file_entry_index": 0}

        @property
        def data_types(self):
            return self._data_types

        @property
        def overlay_methods(self):
            return self._overlay_methods

    monkeypatch.setattr(
        mixed_dataset_module,
        "LArCVDataset",
        lambda **kwargs: DummyDataset(
            {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
                "shared": "tensor",
            },
            {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
                "shared": "first",
            },
        ),
    )
    monkeypatch.setattr(
        mixed_dataset_module,
        "HDF5Dataset",
        lambda **kwargs: DummyDataset(
            {
                "index": "scalar",
                "file_index": "scalar",
                "file_entry_index": "scalar",
                "shared": "tensor",
            },
            {
                "index": "cat",
                "file_index": "cat",
                "file_entry_index": "cat",
                "shared": "last",
            },
        ),
    )

    dataset = MixedDataset(
        larcv={"file_keys": "dummy.root", "schema": {}},
        hdf5={"file_keys": "dummy.h5"},
        dtype="float32",
    )

    with pytest.raises(ValueError, match="overlay collision"):
        _ = dataset.overlay_methods


def test_larcv_dataset_uses_augmenter_and_length(monkeypatch):
    """The LArCV dataset should initialize and apply the configured augmenter."""
    seen = {}

    class DummyParser:
        tree_keys = ["tree_a"]
        returns = "tensor"
        overlay = "cat"

        def __init__(self, dtype):
            self.dtype = dtype

        def __call__(self, data):
            return data["tree_a"]

    class DummyReader:
        entry_index = [0]

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"tree_a": {"value": idx}}

        def get_file_index(self, idx):
            return 0

        def get_file_entry_index(self, idx):
            return idx

        @staticmethod
        def list_data(path):
            return {"dummy": [path]}

    class DummyAugmenter:
        def __call__(self, result):
            result = dict(result)
            result["augmented"] = True
            return result

    def build_augmenter(*, geo=None, **augment):
        seen["geo"] = geo
        seen["augment"] = augment
        return DummyAugmenter()

    monkeypatch.setattr(larcv_dataset_module, "PARSER_DICT", {"dummy": DummyParser})
    monkeypatch.setattr(larcv_dataset_module, "LArCVReader", DummyReader)
    monkeypatch.setattr(dataset_base_module, "AugmentManager", build_augmenter)

    dataset = LArCVDataset(
        schema={"x": {"parser": "dummy"}},
        dtype="float32",
        augment={"mask": {"min_dimensions": [1, 1, 1], "max_dimensions": [1, 1, 1]}},
        geo={"detector": "icarus"},
    )

    assert len(dataset) == 1
    assert dataset[0]["augmented"] is True
    assert seen["geo"] == {"detector": "icarus"}
    assert "mask" in seen["augment"]
    assert dataset.list_data("file.root") == {"dummy": ["file.root"]}


def test_larcv_dataset_parser_failure_logs_and_raises(monkeypatch):
    """The LArCV dataset should log parser failures and re-raise them."""

    class DummyParser:
        tree_keys = ["tree_a"]
        returns = "tensor"
        overlay = "cat"

        def __init__(self, dtype):
            self.dtype = dtype

        def __call__(self, data):
            raise RuntimeError("boom")

    class DummyReader:
        entry_index = [0]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"tree_a": idx}

        def get_file_index(self, idx):
            return 0

        def get_file_entry_index(self, idx):
            return idx

    monkeypatch.setattr(larcv_dataset_module, "PARSER_DICT", {"dummy": DummyParser})
    monkeypatch.setattr(larcv_dataset_module, "LArCVReader", lambda **_: DummyReader())

    dataset = LArCVDataset(schema={"x": {"parser": "dummy"}}, dtype="float32")

    with pytest.raises(RuntimeError, match="boom"):
        dataset[0]


def test_dataset_module_import_safe_without_torch(monkeypatch):
    """The dataset base module should define a stand-in Dataset without torch."""
    import spine.utils.conditional as conditional

    monkeypatch.setattr(conditional, "TORCH_AVAILABLE", False)
    reloaded = importlib.reload(dataset_base_module)
    try:
        assert reloaded.Dataset.__name__ == "Dataset"
    finally:
        monkeypatch.setattr(conditional, "TORCH_AVAILABLE", TORCH_AVAILABLE)
        importlib.reload(dataset_base_module)


def test_dataset_package_exports_classes():
    """The dataset package should export the public dataset classes."""
    assert dataset_module.LArCVDataset is LArCVDataset
    assert dataset_module.HDF5Dataset is HDF5Dataset
