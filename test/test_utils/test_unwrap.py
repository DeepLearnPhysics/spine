"""Test that the batch objects can be unwrapped properly."""

import numpy as np
import pytest

from spine.data import IndexBatch, TensorBatch
from spine.data.list import ObjectList
from spine.io.unwrap import Unwrapper


def _spans(offsets, last=0):
    """Convert cumulative offsets into constructor spans for tests."""
    spans = np.array(offsets, copy=True)
    if len(spans) > 1:
        spans[:-1] = offsets[1:] - offsets[:-1]
    spans[-1] = last
    return spans


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
    index_batch = IndexBatch(indexes, _spans(offsets), counts, single_counts)

    return index_batch


def test_unwrap_multi_volume_index_list(monkeypatch):
    """Test multi-volume index-list batches preserve the nested list structure."""

    class MockTPC:
        num_modules = 2

    class MockGeo:
        tpc = MockTPC()

    monkeypatch.setattr(
        "spine.io.unwrap.GeoManager.get_instance_if_initialized",
        lambda: MockGeo(),
    )

    index_batch = IndexBatch(
        [
            np.array([0, 1]),
            np.array([2]),
            np.array([10]),
            np.array([11, 12]),
        ],
        spans=np.array([10, 0]),
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


def test_unwrap_exports_single_volume_index_spans():
    """Single-volume unwrapping should expose one stored span per entry."""
    orig_index = IndexBatch(
        np.asarray([0, 2, 4], dtype=np.int64),
        spans=np.asarray([5], dtype=np.int64),
        counts=np.asarray([3], dtype=np.int64),
    )

    result = Unwrapper()({"index": [0], "orig_index": orig_index})

    assert result["orig_index_spans"] == [5]
    np.testing.assert_array_equal(result["orig_index"][0], np.asarray([0, 2, 4]))


def test_unwrap_exports_multi_volume_index_spans(monkeypatch):
    """Multi-volume index unwrapping should sum per-volume spans per event."""

    class MockTPC:
        num_modules = 2

    class MockGeo:
        tpc = MockTPC()

    monkeypatch.setattr(
        "spine.io.unwrap.GeoManager.get_instance_if_initialized",
        lambda: MockGeo(),
    )

    orig_index = IndexBatch(
        [
            np.array([0, 1]),
            np.array([2]),
            np.array([10]),
            np.array([11, 12]),
        ],
        spans=np.array([10, 20]),
        counts=np.array([2, 2]),
        single_counts=np.array([2, 1, 1, 2]),
    )

    result = Unwrapper()({"index": [0], "orig_index": orig_index})

    assert result["orig_index_spans"] == [30]
    assert len(result["orig_index"]) == 1
