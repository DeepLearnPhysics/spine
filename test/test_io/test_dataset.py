"""Test that the dataset classes work as intended."""

import importlib

import pytest

from spine.io import dataset as dataset_module
from spine.io.dataset import *
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
    monkeypatch.setattr(dataset_module, "TORCH_AVAILABLE", False)

    with pytest.raises(ImportError, match="PyTorch is required"):
        HDF5Dataset(file_keys=hdf5_data, build_classes=False)


def test_hdf5_dataset_with_augment(monkeypatch, hdf5_data):
    """The HDF5 dataset should apply augmentation when configured."""

    class DummyAugmenter:
        def __call__(self, result):
            result = dict(result)
            result["augmented"] = True
            return result

    monkeypatch.setattr(dataset_module, "AugmentManager", lambda **_: DummyAugmenter())

    dataset = HDF5Dataset(
        file_keys=hdf5_data,
        build_classes=False,
        keys=["run_info"],
        augment={"name": "dummy"},
    )

    assert dataset[0]["augmented"] is True


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

    monkeypatch.setattr(dataset_module, "PARSER_DICT", {"dummy": DummyParser})
    monkeypatch.setattr(dataset_module, "LArCVReader", DummyReader)
    monkeypatch.setattr(dataset_module, "AugmentManager", build_augmenter)

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

    monkeypatch.setattr(dataset_module, "PARSER_DICT", {"dummy": DummyParser})
    monkeypatch.setattr(dataset_module, "LArCVReader", lambda **_: DummyReader())

    dataset = LArCVDataset(schema={"x": {"parser": "dummy"}}, dtype="float32")

    with pytest.raises(RuntimeError, match="boom"):
        dataset[0]


def test_dataset_module_import_safe_without_torch(monkeypatch):
    """The dataset module should define a stand-in Dataset when torch is unavailable."""
    import spine.utils.conditional as conditional

    monkeypatch.setattr(conditional, "TORCH_AVAILABLE", False)
    reloaded = importlib.reload(dataset_module)
    try:
        assert reloaded.Dataset.__name__ == "Dataset"
    finally:
        monkeypatch.setattr(conditional, "TORCH_AVAILABLE", TORCH_AVAILABLE)
        importlib.reload(dataset_module)
