"""Test that the collate function(s) work as intended."""

import numpy as np
import pytest

from mlreco import Meta
from mlreco.io.collate import CollateAll


@pytest.fixture(name='batch_sparse',
                params=[(1, 1), (1, 4), (4, 1), (4, 4)])
def fixture_batch_sparse(request):
    """Generate a batch of typical sparse data from the parsers.

    Returns
    -------
    List[dict]
        One dictionary of data per entry in the batch
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Loop over each entry in the dummy batch
    batch_size = request.param[0]
    num_products = request.param[1]
    batch = []
    for b in range(batch_size):
        # Initialize the entry dictionary
        data = {}

        # Generate a few sparse-type objects
        for name in range(num_products):
            num_points = np.random.randint(low=0, high=100)

            coords = 100*np.random.rand(num_points, 3)
            features = 10*np.random.rand(num_points, 2)
            meta = Meta(lower=[0.,0.,0.], upper=[100.,100.,100.],
                        size=[1.,1.,1.])

            data[f'sparse_{name}'] = (coords, features, meta)

        # Append the batch list
        batch.append(data)

    return batch


@pytest.fixture(name='batch_edge_index',
                params=[(1, 0), (1, 4), (4, 0), (4, 4)])
def fixture_batch_edge_index(request):
    """Generate a batch of typical edge index data from the parsers.

    Returns
    -------
    List[dict]
        One dictionary of data per entry in the batch
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Loop over each entry in the dummy batch
    batch_size = request.param[0]
    num_products = request.param[1]
    batch = []
    for b in range(batch_size):
        # Initialize the entry dictionary
        data = {}

        # Generate a few sparse-type objects
        for name in range(num_products):
            num_edges = np.random.randint(low=0, high=100)

            edge_index = np.random.randint(0, 10, size=(2, num_edges))

            data[f'edge_index_{name}'] = (edge_index, 10)

        # Append the batch list
        batch.append(data)

    return batch


@pytest.mark.parametrize('split, detector', [
    (False, None),
    (True, 'icarus'),
])
def test_collate_sparse(split, detector, batch_sparse):
    """Tests the collation of sparse tensors."""
    # Initialize the collation class
    collate_fn = CollateAll(split=split, detector=detector)

    # Pass the batch through the collate function
    result = collate_fn(batch_sparse)

    # Check that each key in the output if of the same length as the batch.
    # If split into two detector volumes, there should be twice as many
    for k in batch_sparse[0]:
        assert len(result[k]) == len(batch_sparse)*(2**split)


def test_collate_edge_index(batch_edge_index):
    """Tests the collation of edge indexes."""
    # Initialize the collation class
    collate_fn = CollateAll()

    # Pass the batch through the collate function
    result = collate_fn(batch_edge_index)

    # Check that each key in the output if of the same length as the batch
    for k in batch_edge_index[0]:
        assert len(result[k]) == len(batch_edge_index)


def test_collate_scalar():
    """Tests the collation of scalar values."""
    # Initialize the collation class
    collate_fn = CollateAll()

    # Initialize a simple batch of scalars
    batch_scalar = [{'scalar': i} for i in range(4)]

    # Pass the batch through the collate function
    result = collate_fn(batch_scalar)

    # Check that each key in the output if of the same length as the batch
    assert len(result['scalar']) == len(batch_scalar)

    # Check that the input is intact
    for i, data in enumerate(batch_scalar):
        assert data['scalar'] == result['scalar'][i]


def test_collate_list():
    """Tests the collation of simple lists."""
    # Initialize the collation class
    collate_fn = CollateAll()

    # Initialize a simple batch of lists
    batch_list = [{'list': [i]*i} for i in range(4)]

    # Pass the batch through the collate function
    result = collate_fn(batch_list)

    # Check that each key in the output if of the same length as the batch
    assert len(result['list']) == len(batch_list)

    # Check that the input is intact
    for i, data in enumerate(batch_list):
        assert data['list'] == result['list'][i]
