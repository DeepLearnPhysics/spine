"""Test that the loading of data using a full-fledged configuration.

Note: This test requires PyTorch and spine.io.factories which are optional dependencies.
It's excluded from CI core tests and runs only in torch-enabled environments.
"""

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from spine.io import factories as factories_module
from spine.io.factories import collate_factory, dataset_factory, loader_factory
from spine.io.write.csv import CSVWriter
from spine.utils.conditional import TORCH_AVAILABLE

MAX_ITER = 10
MAX_BATCH_ID = MAX_ITER - 1


@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is required for torch-backed dataset factory tests.",
)
def test_dataset_factory_hdf5(hdf5_data):
    """The generic dataset factory should instantiate the HDF5 dataset."""
    dataset = dataset_factory(
        {
            "name": "hdf5",
            "file_keys": hdf5_data,
            "build_classes": False,
            "keys": ["run_info"],
        },
        dtype="float32",
    )

    assert dataset.name == "hdf5"
    assert "run_info" in dataset.data_keys


@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is required for torch-backed dataset factory tests.",
)
def test_dataset_factory_entry_list_warning(hdf5_data):
    """Providing an external entry list should override the config with a warning."""
    cfg = {
        "name": "hdf5",
        "file_keys": hdf5_data,
        "build_classes": False,
    }

    with pytest.warns(UserWarning, match="overwriting"):
        dataset = dataset_factory(cfg, entry_list=[0], dtype="float32")

    assert len(dataset) == 1


@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is required for torch-backed dataset factory tests.",
)
def test_dataset_factory_does_not_forward_none_geo(hdf5_data):
    """Generic dataset construction should not leak `geo=None` into HDF5 readers."""
    dataset = dataset_factory(
        {"name": "hdf5", "file_keys": hdf5_data, "build_classes": False},
        dtype="float32",
        geo=None,
    )

    assert len(dataset) > 0


@pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch is required for collate factory tests."
)
def test_collate_factory():
    """The collate factory should build the generic collate function."""
    collate_fn = collate_factory(
        {"name": "all"},
        data_types={"value": "scalar"},
        overlay_methods={"value": "cat"},
    )

    result = collate_fn([{"value": 1}, {"value": 2}])
    assert result == {"value": [1, 2]}


def test_reader_and_writer_factories(hdf5_data, tmp_path):
    """Reader and writer factories should instantiate the configured class."""
    hdf5_output = tmp_path / "output.csv"
    writer = factories_module.writer_factory({"name": "csv", "file_name": hdf5_output})
    assert writer.name == "csv"

    reader = factories_module.reader_factory(
        {"name": "hdf5", "file_keys": hdf5_data, "build_classes": False}
    )
    assert reader.name == "hdf5"


def test_writer_factory_forwards_prefix_and_split(monkeypatch):
    """Writer factory should only forward explicit prefix/split options."""
    calls = []

    def fake_instantiate(factory_dict, cfg, **kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(factories_module, "instantiate", fake_instantiate)

    factories_module.writer_factory({"name": "csv"})
    factories_module.writer_factory({"name": "csv"}, prefix="input", split=True)

    assert calls[0] == {}
    assert calls[1] == {"prefix": "input", "split": True}


def test_loader_factory_uses_minibatch_and_helpers(monkeypatch):
    """Loader factory should honor minibatch inputs and helper factories."""

    class DummyDataset:
        data_types = {"value": "scalar"}
        overlay_methods = {"value": "cat"}

    captured = {}

    class DummyDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured.update(kwargs)

    fake_torch = types.ModuleType("torch")
    fake_utils = types.ModuleType("torch.utils")
    fake_data = types.ModuleType("torch.utils.data")
    fake_data.DataLoader = DummyDataLoader
    fake_utils.data = fake_data
    fake_torch.utils = fake_utils

    monkeypatch.setattr(factories_module, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        factories_module, "dataset_factory", lambda *args, **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        factories_module, "sampler_factory", lambda *args, **kwargs: "sampler"
    )
    monkeypatch.setattr(
        factories_module, "collate_factory", lambda *args, **kwargs: "collate"
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_data)

    factories_module.loader_factory(
        dataset={"name": "dummy"},
        dtype="float32",
        minibatch_size=2,
        sampler={"name": "sequential"},
        collate_fn={"name": "all"},
        pin_memory=True,
    )

    assert isinstance(captured["dataset"], DummyDataset)
    assert captured["batch_size"] == 2
    assert captured["sampler"] == "sampler"
    assert captured["collate_fn"] == "collate"
    assert captured["pin_memory"] is True


def test_dataset_factory_forwards_geo(monkeypatch):
    """Dataset factory should only forward geometry when it is provided."""
    seen = []

    monkeypatch.setattr(
        factories_module, "module_dict", lambda module: {"dummy": object()}
    )

    def fake_instantiate(factory_dict, cfg, **kwargs):
        seen.append(kwargs)
        return kwargs

    monkeypatch.setattr(factories_module, "instantiate", fake_instantiate)

    factories_module.dataset_factory({"name": "dummy"}, dtype="float32")
    factories_module.dataset_factory(
        {"name": "dummy"}, dtype="float32", geo={"detector": "icarus"}
    )

    assert seen[0] == {"dtype": "float32"}
    assert seen[1] == {"dtype": "float32", "geo": {"detector": "icarus"}}


@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is required for distributed sampler factory tests.",
)
def test_sampler_factory_wraps_distributed(monkeypatch):
    """Sampler factory should wrap the sampler in distributed mode."""
    import spine.io.sample as sample_module

    wrapped = {}

    class DummySampler:
        batch_size = 4

    monkeypatch.setattr(
        factories_module, "instantiate", lambda *args, **kwargs: DummySampler()
    )
    monkeypatch.setattr(
        sample_module,
        "DistributedProxySampler",
        lambda sampler, num_replicas, rank: wrapped.setdefault(
            "value", (sampler, num_replicas, rank)
        ),
    )

    result = factories_module.sampler_factory(
        {"name": "dummy"},
        dataset=[1, 2],
        minibatch_size=4,
        distributed=True,
        num_replicas=2,
        rank=1,
    )

    assert wrapped["value"][1:] == (2, 1)
    assert result == wrapped["value"]


def test_loader_factory_rejects_missing_torch(monkeypatch):
    """The loader factory should fail clearly when torch is unavailable."""
    monkeypatch.setattr(factories_module, "TORCH_AVAILABLE", False)

    with pytest.raises(ImportError, match="PyTorch is required"):
        loader_factory(dataset={}, dtype="float32", batch_size=1)


def test_sampler_factory_rejects_missing_torch(monkeypatch):
    """The sampler factory should fail clearly when torch is unavailable."""
    monkeypatch.setattr(factories_module, "TORCH_AVAILABLE", False)

    with pytest.raises(ImportError, match="PyTorch is required"):
        factories_module.sampler_factory({}, dataset=[], minibatch_size=1)


@pytest.mark.parametrize("cfg_file", ["test_loader.cfg"])
@pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch is required for loader factory tests."
)
def test_loader(cfg_file, larcv_data, quiet=True, csv=False):
    """Tests the loading of data using a full IO configuration."""
    # Fetch the configuration
    cfg_path = Path(cfg_file)
    if not cfg_path.is_file():
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "config" / cfg_file
            if candidate.is_file():
                cfg_path = candidate
                break
    if not cfg_path.is_file():
        raise ValueError(f"Configuration file not found: {cfg_file}")

    # If requested, intialize a CSV output
    if csv:
        csv = CSVWriter("test.csv")

    # Initialize the loader
    with open(cfg_path, "r", encoding="utf-8") as cfg_str:
        # Load configuration dictionary
        cfg = yaml.safe_load(cfg_str)

        # Update the path to the file
        cfg["io"]["loader"]["dataset"]["file_keys"] = larcv_data

    loader = loader_factory(dtype="float32", **cfg["io"]["loader"])

    # Loop
    tstart = time.time()
    tsum = 0.0
    t0 = 0.0
    for batch_id, data in enumerate(loader):
        titer = time.time() - tstart
        if not quiet:
            print("Batch", batch_id)
            for key, value in data.items():
                print("   ", key, np.shape(value))
            print("Duration", titer, "[s]")
        if batch_id < 1:
            t0 = titer
        tsum += titer
        if csv:
            csv.append({"iter": batch_id, "t": titer})
        if (batch_id + 1) == MAX_ITER:
            break
        tstart = time.time()

    if not quiet:
        print("Total time:", tsum, "[s] ... Average time:", tsum / MAX_BATCH_ID, "[s]")
        if MAX_BATCH_ID > 1:
            print(
                "First iter:",
                t0,
                "[s] ... Average w/o first iter:",
                (tsum - t0) / (MAX_BATCH_ID - 1),
                "[s]",
            )
