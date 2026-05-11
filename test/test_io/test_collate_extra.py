"""Additional tests for uncovered collate branches."""

import numpy as np
import pytest

from spine.data import EdgeIndexBatch, IndexBatch, Meta, TensorBatch
from spine.io.collate import CollateAll
from spine.io.parse.data import ParserTensor
from spine.utils.conditional import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch is required for collate tests."
)


def make_meta():
    """Build a simple metadata object."""
    return Meta(
        lower=np.asarray([0.0, 0.0, 0.0]),
        upper=np.asarray([10.0, 10.0, 10.0]),
        size=np.asarray([1.0, 1.0, 1.0]),
        count=np.asarray([10, 10, 10]),
    )


def test_collate_with_overlay():
    """CollateAll should apply overlay before batching."""
    collate_fn = CollateAll(
        data_types={"run": "scalar"},
        overlay={"multiplicity": 2},
        overlay_methods={"run": "match"},
    )

    result = collate_fn([{"run": 1}, {"run": 1}])
    assert result == {"run": [1]}


def test_collate_index_tensor_returns_index_batch():
    """One-dimensional index tensors should produce an IndexBatch."""
    batch = [
        {"index_tensor": ParserTensor(features=np.asarray([0, 1]), global_shift=2)},
        {"index_tensor": ParserTensor(features=np.asarray([0, 2]), global_shift=3)},
    ]
    collate_fn = CollateAll(data_types={"index_tensor": "tensor"})

    result = collate_fn(batch)
    assert isinstance(result["index_tensor"], IndexBatch)


def test_collate_edge_index_tensor_returns_edge_index_batch():
    """Two-dimensional index tensors should produce an EdgeIndexBatch."""
    batch = [
        {
            "edge_tensor": ParserTensor(
                features=np.asarray([[0, 1], [1, 0]]), global_shift=2
            )
        },
        {
            "edge_tensor": ParserTensor(
                features=np.asarray([[0, 1], [1, 0]]), global_shift=2
            )
        },
    ]
    collate_fn = CollateAll(data_types={"edge_tensor": "tensor"})

    result = collate_fn(batch)
    assert isinstance(result["edge_tensor"], EdgeIndexBatch)


def test_collate_index_list_tensor_returns_index_batch():
    """List-backed index tensors should produce an IndexBatch with per-index sizes."""
    batch = [
        {
            "index_tensor": ParserTensor(
                features=[np.asarray([0, 2]), np.asarray([1])],
                global_shift=3,
                single_counts=np.asarray([2, 1]),
            )
        },
        {
            "index_tensor": ParserTensor(
                features=[np.asarray([0, 1, 2])],
                global_shift=3,
            )
        },
    ]
    collate_fn = CollateAll(data_types={"index_tensor": "tensor"})

    result = collate_fn(batch)
    assert isinstance(result["index_tensor"], IndexBatch)
    assert result["index_tensor"].counts.tolist() == [2, 1]
    assert result["index_tensor"].single_counts.tolist() == [2, 1, 3]


def test_collate_feature_tensors_without_coords():
    """Feature-only tensors should be collated with stack_feat_tensors."""
    batch = [
        {
            "feat": ParserTensor(
                features=np.asarray([[1.0], [2.0]], dtype=np.float32), feats_only=True
            )
        },
        {
            "feat": ParserTensor(
                features=np.asarray([[3.0]], dtype=np.float32), feats_only=True
            )
        },
    ]
    collate_fn = CollateAll(data_types={"feat": "tensor"})

    result = collate_fn(batch)
    assert isinstance(result["feat"], TensorBatch)
    assert len(result["feat"]) == 2


def test_collate_split_feature_tensors_with_source(monkeypatch):
    """Split feature collation should use the provided source module mapping."""

    class DummyTPC:
        num_modules = 2

    class DummyGeo:
        tpc = DummyTPC()

    monkeypatch.setattr("spine.io.collate.GeoManager.get_instance", lambda: DummyGeo())

    batch = [
        {
            "feat": ParserTensor(
                features=np.asarray([[10.0], [20.0]], dtype=np.float32), feats_only=True
            ),
            "source": ParserTensor(
                features=np.asarray([[0], [1]], dtype=np.int64), feats_only=True
            ),
        }
    ]
    collate_fn = CollateAll(
        data_types={"feat": "tensor"},
        split=True,
        source={"feat": "source"},
    )

    result = collate_fn(batch)
    assert isinstance(result["feat"], TensorBatch)
    assert len(result["feat"]) == 2


def test_collate_split_coordinate_tensor_multi_point(monkeypatch):
    """Split coordinate collation should handle rows with multiple points."""

    class DummyTPC:
        num_modules = 2

    class DummyGeo:
        tpc = DummyTPC()

        @staticmethod
        def split(coords, target_id, meta=None):
            return coords, [np.asarray([0, 2]), np.asarray([1, 3])]

    monkeypatch.setattr("spine.io.collate.GeoManager.get_instance", lambda: DummyGeo())

    tensor = ParserTensor(
        coords=np.asarray(
            [
                [0, 0, 0, 1, 1, 1],
                [2, 2, 2, 3, 3, 3],
            ],
            dtype=np.float32,
        ),
        features=np.asarray([[1.0], [2.0]], dtype=np.float32),
        meta=make_meta(),
    )
    collate_fn = CollateAll(data_types={"voxels": "tensor"}, split=True)

    result = collate_fn([{"voxels": tensor}])
    assert isinstance(result["voxels"], TensorBatch)
    assert len(result["voxels"]) == 2
