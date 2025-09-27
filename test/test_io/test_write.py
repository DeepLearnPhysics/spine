"""Test that the writer classes work as intended."""

import os

import h5py
import numpy as np
import pytest

from spine.data import (CRTHit, Flash, Meta, Neutrino, ObjectList, Particle,
                        RunInfo, Trigger)
from spine.io.core.write import *


@pytest.fixture(name="hdf5_output")
def fixture_hdf5_output(tmp_path):
    """Create a dummy output path for an HDF5 file.

    Parameters
    ----------
    tmp_path : str
       Generic pytest fixture used to handle temporary test files
    """
    return os.path.join(tmp_path, "dummy.h5")


@pytest.fixture(name="tensor_list")
def fixture_tensor_list(request):
    """Generates a dummy list of unwrapped tensors."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of tensors of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    tensors = []
    for i, s in enumerate(sizes):
        tensors.append(np.random.rand(s, 5))

    return tensors


@pytest.fixture(name="index_list")
def fixture_index_list(request):
    """Generates a dummy list of unwrapped index lists."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of index lists of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    indexes = []
    default = np.empty(0, dtype=np.int64)
    for i, s in enumerate(sizes):
        index = np.arange(s)
        if s > 1:
            index = np.split(index, [np.random.randint(1, s)])
            indexes.append(index)
        elif s == 1:
            indexes.append([index])
        else:
            indexes.append(ObjectList(index, default))

    return indexes


@pytest.fixture(name="edge_index_list")
def fixture_edge_index_list(request):
    """Generates a dummy list of unwrapped edge indexes."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of index lists of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    edge_indexes = []
    for i, s in enumerate(sizes):
        edge_indexes.append(np.random.rand(s, 2))

    return edge_indexes


@pytest.mark.parametrize(
    "tensor_list, index_list, edge_index_list",
    [(0, 0, 0), (10, 10, 25), ((0, 0), (0, 0), (0, 0)), ((5, 10), (5, 10), (8, 25))],
    indirect=True,
)
def test_hdf5_writer(hdf5_output, tensor_list, index_list, edge_index_list):
    """Tests the HDF5 writer."""
    # Create an output similar to that of the full chain
    batch_size = len(tensor_list)
    sizes = [len(t) for t in tensor_list]
    data = {
        "index": np.arange(batch_size),
        "dummy_data": tensor_list,
        "dummy_meta": [Meta()] * batch_size,
        "dummy_run_info": [RunInfo()] * batch_size,
        "dummy_trigger": [Trigger()] * batch_size,
        "dummy_particles": generate_object_list(Particle, sizes),
        "dummy_neutrinos": generate_object_list(Neutrino, sizes),
        "dummy_flashes": generate_object_list(Flash, sizes),
        "dummy_crthits": generate_object_list(CRTHit, sizes),
        "dummy_tensor": tensor_list,
        "dummy_clusts": index_list,
        "dummy_edge_index": edge_index_list,
    }

    # Initialize the writer
    writer = HDF5Writer(hdf5_output)

    # Write output
    writer(data)


def generate_object_list(cls, sizes):
    """Generates a dummy list of lists of objects of the request class.

    Parameters
    ----------
    cls : object
        Class that the objects should belong to
    sizes : List[int]
        Number of objects in each list

    Returns
    -------
    List[ObjectList[obj]]
        List of typed lists of objects
    """
    return [ObjectList([cls() for _ in range(s)], cls()) for s in sizes]
