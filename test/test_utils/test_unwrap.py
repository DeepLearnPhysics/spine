"""Test that the batch objects can be unwrapped properly."""

import numpy as np
import pytest

from spine.data import IndexBatch, TensorBatch
from spine.data.list import ObjectList
from spine.utils.unwrap import Unwrapper


@pytest.fixture(name="tensor_batch")
def fixture_tensor_batch(request):
    """Generates a dummy tensor batch."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of tensors of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    batch_size = len(sizes)
    tensors = []
    for i, s in enumerate(sizes):
        tensors.append(np.random.rand(s, 5))

    # Initialize the batch object
    tensor_batch = TensorBatch.from_list(tensors)

    return tensor_batch


@pytest.fixture(name="index_batch")
def fixture_index_batch(request):
    """Generates a dummy index batch."""
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Generate the request number of tensors of a predeterminate size
    sizes = request.param
    if np.isscalar(sizes):
        sizes = [sizes]

    batch_size = len(sizes)
    indexes = []
    offsets, counts, single_counts = [], [], []
    offset = 0
    for i, s in enumerate(sizes):
        index = offset + np.arange(s)
        offsets.append(offset)
        offset += 2 * len(index)
        if s > 1:
            index = np.split(index, [np.random.randint(1, s - 1)])
            indexes.extend(index)
            counts.append(2)
            single_counts.extend([len(c) for c in index])
        else:
            counts.append(0)

    # Initialize the batch objec
    index_batch = IndexBatch(indexes, offsets, counts, single_counts)

    return index_batch


def test_unwrap_multi_volume_index_list(monkeypatch):
    """Test multi-volume index-list batches preserve the nested list structure."""

    class MockTPC:
        num_modules = 2

    class MockGeo:
        tpc = MockTPC()

    monkeypatch.setattr(
        "spine.utils.unwrap.GeoManager.get_instance_if_initialized",
        lambda: MockGeo(),
    )

    index_batch = IndexBatch(
        [
            np.array([0, 1]),
            np.array([2]),
            np.array([10]),
            np.array([11, 12]),
        ],
        offsets=np.array([0, 10]),
        counts=np.array([2, 2]),
        single_counts=np.array([2, 1, 1, 2]),
    )

    unwrapper = Unwrapper()
    result = unwrapper._unwrap_index(index_batch)

    assert len(result) == 1
    assert isinstance(result[0], ObjectList)
    assert len(result[0]) == 4
    np.testing.assert_array_equal(result[0][0], np.array([0, 1]))
    np.testing.assert_array_equal(result[0][1], np.array([2]))
    np.testing.assert_array_equal(result[0][2], np.array([10]))
    np.testing.assert_array_equal(result[0][3], np.array([11, 12]))
