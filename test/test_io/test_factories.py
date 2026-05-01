"""Test that the loading of data using a full-fledged configuration.

Note: This test requires PyTorch and spine.io.factories which are optional dependencies.
It's excluded from CI core tests and runs only in torch-enabled environments.
"""

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from spine.io.factories import dataset_factory, loader_factory
from spine.io.write.csv import CSVWriter

MAX_ITER = 10
MAX_BATCH_ID = MAX_ITER - 1


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


@pytest.mark.parametrize("cfg_file", ["test_loader.cfg"])
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
