"""Test that the batch objects can be unwrapped properly."""

import numpy as np
import pytest

from spine.data import IndexBatch, TensorBatch
from spine.data.batch.edge_index import EdgeIndexBatch
from spine.data.list import ObjectList
from spine.io.unwrap import Unwrapper


def _spans(offsets, last=0):
    """Convert cumulative offsets into constructor spans for tests."""
    spans = np.array(offsets, copy=True)
    if len(spans) > 1:
        spans[:-1] = offsets[1:] - offsets[:-1]
    spans[-1] = last
    return spans


@pytest.fixture(autouse=True)
def fixture_no_global_geo(monkeypatch):
    """Keep unwrap tests independent of geometry initialized by other tests."""
    monkeypatch.setattr(
        "spine.io.unwrap.GeoManager.get_instance_if_initialized",
        lambda: None,
    )


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


def test_unwrap_tensor_batch_single_volume():
    tensor = np.arange(10).reshape(2, 5)
    batch = TensorBatch(tensor, counts=[2])
    unwrapper = Unwrapper()
    result = unwrapper._unwrap_tensor(batch)
    assert isinstance(result, list)
    assert np.allclose(result[0], tensor)


def test_unwrap_tensor_batch_remove_batch_col():
    tensor = np.arange(12).reshape(2, 6)
    batch = TensorBatch(tensor, counts=[2])
    batch.has_batch_col = True
    unwrapper = Unwrapper(remove_batch_col=True)
    result = unwrapper._unwrap_tensor(batch)
    assert all(r.shape[1] == 5 for r in result)


def test_unwrap_tensor_batch_list():
    """A list of tensor batches should unwrap to per-entry tensor lists."""
    batch_1 = TensorBatch(np.array([[1, 2], [3, 4]]), counts=[1, 1])
    batch_2 = TensorBatch(np.array([[5, 6], [7, 8]]), counts=[1, 1])

    result = Unwrapper()({"index": [0, 1], "tensors": [batch_1, batch_2]})

    assert len(result["tensors"]) == 2
    np.testing.assert_array_equal(result["tensors"][0][0], np.array([[1, 2]]))
    np.testing.assert_array_equal(result["tensors"][0][1], np.array([[5, 6]]))


def test_unwrap_tensor_batch_convertible_values():
    """Portable model outputs unwrap directly and inside feature-map lists."""

    class Convertible:
        def __init__(self, values):
            self.values = np.asarray(values)

        def to_tensor_batch(self):
            return TensorBatch(self.values, counts=[1, 1])

    direct = Convertible([[1, 2], [3, 4]])
    levels = [
        Convertible([[5, 6], [7, 8]]),
        Convertible([[9, 10], [11, 12]]),
    ]

    result = Unwrapper()(
        {
            "index": [0, 1],
            "direct": direct,
            "levels": levels,
        }
    )

    np.testing.assert_array_equal(result["direct"][0], np.array([[1, 2]]))
    np.testing.assert_array_equal(result["levels"][1][0], np.array([[7, 8]]))
    np.testing.assert_array_equal(result["levels"][1][1], np.array([[11, 12]]))


def test_unwrap_tensor_batch_multi_volume(monkeypatch):
    class MockTPC:
        num_modules = 2

    class MockGeo:
        tpc = MockTPC()

        def translate(self, x, *a, **k):
            return x + 10

    monkeypatch.setattr(
        "spine.io.unwrap.GeoManager.get_instance_if_initialized", lambda: MockGeo()
    )
    tensor = np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 2],
            [0, 2, 2, 2, 3],
            [1, 3, 3, 3, 4],
        ],
        dtype=np.float32,
    )
    batch = TensorBatch(
        tensor,
        counts=[1, 1, 1, 1],
        has_batch_col=True,
        coord_cols=np.array([1, 2, 3]),
    )
    meta = [type("Meta", (), {"size": 1})(), type("Meta", (), {"size": 1})()]

    result = Unwrapper(remove_batch_col=True)(
        {"index": [0, 1], "meta": meta, "points": batch}
    )

    assert len(result["points"]) == 2
    assert result["points"][0].shape == (2, 4)
    np.testing.assert_array_equal(result["points"][0][0], np.array([0, 0, 0, 1]))
    np.testing.assert_array_equal(result["points"][0][1], np.array([11, 11, 11, 2]))


def test_unwrap_tensor_batch_multi_volume_requires_numpy(monkeypatch):
    """Multi-volume coordinate translation requires numpy-backed tensor entries."""

    class MockTPC:
        num_modules = 2

    class MockGeo:
        tpc = MockTPC()

        def translate(self, x, *a, **k):
            return x

    monkeypatch.setattr(
        "spine.io.unwrap.GeoManager.get_instance_if_initialized", lambda: MockGeo()
    )
    monkeypatch.setattr(TensorBatch, "__getitem__", lambda self, index: object())

    batch = TensorBatch(
        np.ones((4, 5), dtype=np.float32),
        counts=[1, 1, 1, 1],
        coord_cols=np.array([1, 2, 3]),
    )
    unwrapper = Unwrapper()
    unwrapper.batch_size = 2
    meta = [type("Meta", (), {"size": 1})(), type("Meta", (), {"size": 1})()]

    with pytest.raises(TypeError, match="numpy-backed"):
        unwrapper._unwrap_tensor(batch, meta)


def test_unwrap_index_edgeindexbatch_single_volume():
    edges = np.array([[0, 1], [1, 2]])
    batch = EdgeIndexBatch(edges, counts=[2], spans=[2], directed=True)
    unwrapper = Unwrapper()
    result = unwrapper._unwrap_index(batch)
    assert isinstance(result, list)
    assert np.allclose(result[0], edges)


def test_unwrap_index_edgeindexbatch_multi_volume(monkeypatch):
    class MockTPC:
        num_modules = 2

    class MockGeo:
        tpc = MockTPC()

    monkeypatch.setattr(
        "spine.io.unwrap.GeoManager.get_instance_if_initialized", lambda: MockGeo()
    )
    edges = np.array(
        [
            [0, 10, 100, 200],
            [1, 11, 101, 201],
        ],
        dtype=np.int64,
    )
    batch = EdgeIndexBatch(
        edges, counts=[1, 1, 1, 1], spans=[10, 20, 30, 40], directed=True
    )

    result = Unwrapper()({"index": [0, 1], "edge_index": batch})

    assert len(result["edge_index"]) == 2
    np.testing.assert_array_equal(result["edge_index"][0], np.array([[0, 1], [10, 11]]))
    assert result["edge_index_spans"] == [30, 70]


def test_unwrap_index_spans_edgeindexbatch():
    edges = np.array([[0, 1], [1, 2]])
    batch = EdgeIndexBatch(edges, counts=[2], spans=[2], directed=True)
    unwrapper = Unwrapper()
    result = unwrapper._unwrap_index_spans(batch)
    assert result == [2]


def test_unwrap_index_spans_edgeindexbatch_multi_volume(monkeypatch):
    class MockTPC:
        num_modules = 2

    class MockGeo:
        tpc = MockTPC()

    monkeypatch.setattr(
        "spine.io.unwrap.GeoManager.get_instance_if_initialized", lambda: MockGeo()
    )
    edges = np.array([[0, 10, 100, 200], [1, 11, 101, 201]], dtype=np.int64)
    batch = EdgeIndexBatch(
        edges, counts=[1, 1, 1, 1], spans=[4, 6, 8, 10], directed=True
    )
    unwrapper = Unwrapper()
    unwrapper.batch_size = 2
    result = unwrapper._unwrap_index_spans(batch)
    assert result == [10, 18]


def test_unwrap_index_spans_tensor_like():
    """Span unwrapping should accept tensor-like objects with numpy conversion."""

    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([3, 5], dtype=np.int64)

    class BatchLike:
        spans = TensorLike()
        batch_size = 2

    result = Unwrapper()._unwrap_index_spans(BatchLike())

    assert result == [3, 5]


def test_unwrap_empty_list_raises():
    unwrapper = Unwrapper()
    with pytest.raises(ValueError, match="empty list"):
        unwrapper._unwrap("foo", [], None)


def test_unwrap_batch_size_none_raises():
    unwrapper = Unwrapper()
    unwrapper.batch_size = None
    with pytest.raises(ValueError, match="Batch size should be set"):
        unwrapper._unwrap("foo", np.array([1, 2]), None)


def test_unwrap_invalid_type_raises():
    unwrapper = Unwrapper()
    unwrapper.batch_size = 1
    with pytest.raises(ValueError, match="not unwrappable"):
        unwrapper._unwrap("foo", object(), None)


def test_unwrap_tensor_batch_multi_volume_missing_geo():
    tensor = np.arange(20).reshape(4, 5)
    batch = TensorBatch(tensor, counts=[2, 2])
    batch.batch_size = 4
    unwrapper = Unwrapper()
    unwrapper.geo = None
    unwrapper.num_volumes = 2
    meta = [type("Meta", (), {"size": 1})(), type("Meta", (), {"size": 1})()]
    with pytest.raises(ValueError, match="Geometry must be initialized"):
        unwrapper._unwrap_tensor(batch, meta)


def test_unwrap_tensor_batch_multi_volume_missing_meta(monkeypatch):
    class MockTPC:
        num_modules = 2

    class MockGeo:
        tpc = MockTPC()

        def translate(self, x, *a, **k):
            return x

    monkeypatch.setattr(
        "spine.io.unwrap.GeoManager.get_instance_if_initialized", lambda: MockGeo()
    )
    tensor = np.arange(20).reshape(4, 5)
    batch = TensorBatch(tensor, counts=[2, 2])
    batch.batch_size = 4
    unwrapper = Unwrapper()
    unwrapper.geo = MockGeo()
    unwrapper.num_volumes = 2
    with pytest.raises(ValueError, match="Metadata must be provided"):
        unwrapper._unwrap_tensor(batch, None)
