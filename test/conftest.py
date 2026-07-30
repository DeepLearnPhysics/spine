"""Sets up fixtures general to the entire test suite of this package.

This file is read during the collection phase of pytest when running anything
inside this directory.
"""

import os
import urllib
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
MODEL_TEST_DIR = Path(__file__).resolve().parent / "test_model"


def pytest_addoption(parser):
    """Defines testing command line arguments that can be passed to any
    test scripts inside the general test directory.
    """
    # Optional command line argument to specify one or several image sizes
    parser.addoption(
        "--N",
        type=int,
        nargs="+",
        action="store",
        default=[192],
        help="Image size (default: 192)",
    )
    parser.addoption(
        "--model-tests",
        action="store_true",
        default=False,
        help="Collect tests that require the full model dependency stack.",
    )

    # Adds an option to the pytest.ini file
    parser.addini(
        "larcv_datafile", "URL to small LArCV data file for testing.", type="linelist"
    )
    parser.addini(
        "hdf5_datafile", "URL to small HDF5 file for testing.", type="linelist"
    )


def pytest_ignore_collect(collection_path, config):
    """Keep the full-dependency model suite opt-in."""
    path = Path(collection_path).resolve()
    if path == MODEL_TEST_DIR or MODEL_TEST_DIR in path.parents:
        return not config.getoption("--model-tests")
    return None


def pytest_generate_tests(metafunc):
    """Appends general parameters to all tests."""
    # If a test requires the fixture N, use the command line option.
    if "N" in metafunc.fixturenames:
        metafunc.parametrize("N", metafunc.config.getoption("--N"))

    # If a test requires the fixture datafile use the command line option.
    if "larcv_datafile" in metafunc.fixturenames:
        metafunc.parametrize(
            "larcv_datafile",
            metafunc.config.getini("larcv_datafile"),
            scope="session",
        )
    if "hdf5_datafile" in metafunc.fixturenames:
        metafunc.parametrize(
            "hdf5_datafile",
            metafunc.config.getini("hdf5_datafile"),
            scope="session",
        )


@pytest.fixture(name="larcv_data", scope="session")
def fixture_larcv_data(tmp_path_factory, larcv_datafile):
    """Download a LArCV ROOT datafile here and cache it.

    Parameters
    ----------
    tmp_path : str
       Generic pytest fixture used to handle temporary test files
    larcv_datafile : str
       Name of the datafile to pull (default defined in pytest.ini)
    """
    filename = "test"
    datafile_url = larcv_datafile
    data_path = os.path.join(tmp_path_factory.mktemp("larcv_data"), filename + ".root")
    urllib.request.urlretrieve(datafile_url, data_path)

    return data_path


@pytest.fixture(name="hdf5_data", scope="session")
def fixture_hdf5_data(tmp_path_factory, hdf5_datafile):
    """Download an HDF5 datafile here and cache it.

    Parameters
    ----------
    tmp_path : str
       Generic pytest fixture used to handle temporary test files
    hdf5_datafile : str
       Name of the datafile to pull (default defined in pytest.ini)
    """
    filename = "test"
    datafile_url = hdf5_datafile
    data_path = os.path.join(tmp_path_factory.mktemp("hdf5_data"), filename + ".h5")
    urllib.request.urlretrieve(datafile_url, data_path)

    return data_path
