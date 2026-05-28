"""Test that the collate function(s) work as intended."""

import numpy as np
import pytest

from spine.data import EdgeIndexBatch, IndexBatch, Meta, TensorBatch
from spine.geo import GeoManager
from spine.io.collate import CollateAll
from spine.io.parse.data import (
    ParserEdgeIndex,
    ParserIndex,
    ParserIndexList,
    ParserTensor,
)


def make_meta():
    """Build a simple metadata object."""
    return Meta(
        lower=np.asarray([0.0, 0.0, 0.0]),
        upper=np.asarray([10.0, 10.0, 10.0]),
        size=np.asarray([1.0, 1.0, 1.0]),
        count=np.asarray([10, 10, 10]),
    )


@pytest.fixture(name="batch_sparse", params=[(1, 1), (1, 4), (4, 1), (4, 4)])
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

            coords = 100 * np.random.rand(num_points, 3)
            features = 10 * np.random.rand(num_points, 2)
            meta = Meta(
                lower=np.asarray([0.0, 0.0, 0.0]),
                upper=np.asarray([100.0, 100.0, 100.0]),
                size=np.asarray([1.0, 1.0, 1.0]),
                count=np.asarray([100, 100, 100]),
            )

            data[f"sparse_{name}"] = ParserTensor(
                coords=coords, features=features, meta=meta
            )

        # Append the batch list
        batch.append(data)

    return batch


@pytest.fixture(name="batch_edge_index", params=[(1, 0), (1, 4), (4, 0), (4, 4)])
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

            data[f"edge_index_{name}"] = ParserEdgeIndex(
                features=edge_index, global_shift=10
            )

        # Append the batch list
        batch.append(data)

    return batch


@pytest.mark.parametrize(
    "split, detector",
    [
        (False, None),
        (True, "icarus"),
    ],
)
def test_collate_sparse(split, detector, batch_sparse):
    """Tests the collation of sparse tensors."""
    # Initialize the geoemtry for the test, if needed
    if detector:
        GeoManager.initialize_or_get(detector=detector)

    # Initialize the collation class
    collate_fn = CollateAll(
        data_types={key: "tensor" for key in batch_sparse[0].keys()}, split=split
    )

    # Pass the batch through the collate function
    result = collate_fn(batch_sparse)

    # Check that each key in the output if of the same length as the batch.
    # If split into two detector volumes, there should be twice as many
    for k in batch_sparse[0]:
        assert len(result[k]) == len(batch_sparse) * (2**split)


def test_collate_edge_index(batch_edge_index):
    """Tests the collation of edge indexes."""
    # Initialize the collation class
    collate_fn = CollateAll(
        data_types={key: "tensor" for key in batch_edge_index[0].keys()}
    )

    # Pass the batch through the collate function
    result = collate_fn(batch_edge_index)

    # Check that each key in the output if of the same length as the batch
    for k in batch_edge_index[0]:
        assert len(result[k]) == len(batch_edge_index)


def test_collate_scalar():
    """Tests the collation of scalar values."""
    # Initialize the collation class
    collate_fn = CollateAll(data_types={"scalar": "scalar"})

    # Initialize a simple batch of scalars
    batch_scalar = [{"scalar": i} for i in range(4)]

    # Pass the batch through the collate function
    result = collate_fn(batch_scalar)

    # Check that each key in the output if of the same length as the batch
    assert len(result["scalar"]) == len(batch_scalar)

    # Check that the input is intact
    for i, data in enumerate(batch_scalar):
        assert data["scalar"] == result["scalar"][i]


def test_collate_list():
    """Tests the collation of simple lists."""
    # Initialize the collation class
    collate_fn = CollateAll(data_types={"list": "list"})

    # Initialize a simple batch of lists
    batch_list = [{"list": [i] * i} for i in range(4)]

    # Pass the batch through the collate function
    result = collate_fn(batch_list)

    # Check that each key in the output if of the same length as the batch
    assert len(result["list"]) == len(batch_list)

    # Check that the input is intact
    for i, data in enumerate(batch_list):
        assert data["list"] == result["list"][i]


def test_collate_tensor_dispatch_errors():
    """Tensor dispatch should reject unsupported payloads."""
    collate_fn = CollateAll(data_types={"bad": "tensor"})

    with pytest.raises(TypeError, match="Unsupported parser payload type"):
        collate_fn([{"bad": object()}])


def test_collate_coordinate_tensor_without_split():
    """Coordinate tensors should stack with batch ids and coordinates."""
    meta = Meta(
        lower=np.asarray([0.0, 0.0, 0.0]),
        upper=np.asarray([10.0, 10.0, 10.0]),
        size=np.asarray([1.0, 1.0, 1.0]),
        count=np.asarray([10, 10, 10]),
    )
    batch = [
        {
            "voxels": ParserTensor(
                coords=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
                features=np.asarray([[1.0], [2.0]], dtype=np.float32),
                meta=meta,
            )
        },
        {
            "voxels": ParserTensor(
                coords=np.asarray([[2, 2, 2]], dtype=np.int64),
                features=np.asarray([[3.0]], dtype=np.float32),
                meta=meta,
            )
        },
    ]
    result = CollateAll(data_types={"voxels": "tensor"})(batch)

    tensor = result["voxels"]
    assert isinstance(tensor, TensorBatch)
    assert tensor.counts.tolist() == [2, 1]
    assert tensor.tensor.shape == (3, 5)


def test_collate_index_tensor_and_edge_tensor_offsets():
    """Index-like tensors should be offset and wrapped in the right batch type."""
    collate_fn = CollateAll(data_types={"flat": "tensor", "edge": "tensor"})
    batch = [
        {
            "flat": ParserIndex(
                features=np.asarray([0, 1], dtype=np.int64), global_shift=2
            ),
            "edge": ParserEdgeIndex(
                features=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
                global_shift=2,
            ),
        },
        {
            "flat": ParserIndex(
                features=np.asarray([0], dtype=np.int64), global_shift=1
            ),
            "edge": ParserEdgeIndex(
                features=np.asarray([[0], [0]], dtype=np.int64),
                global_shift=1,
            ),
        },
    ]
    result = collate_fn(batch)

    assert isinstance(result["flat"], IndexBatch)
    assert result["flat"].index.tolist() == [0, 1, 2]
    assert isinstance(result["edge"], EdgeIndexBatch)
    assert result["edge"].index.tolist() == [[0, 1, 2], [1, 0, 2]]


def test_collate_with_overlay():
    """CollateAll should apply overlay before batching."""
    collate_fn = CollateAll(
        data_types={"run": "scalar"},
        overlay={"multiplicity": 2},
        overlay_methods={"run": "match"},
    )

    result = collate_fn([{"run": 1}, {"run": 1}])
    assert result == {"run": [1]}


def test_collate_overlay_requires_overlay_methods():
    """CollateAll should require overlay methods when overlaying is enabled."""
    with pytest.raises(ValueError, match="overlay_methods"):
        CollateAll(data_types={"run": "scalar"}, overlay={"multiplicity": 2})


def test_collate_index_tensor_returns_index_batch():
    """One-dimensional index tensors should produce an IndexBatch."""
    batch = [
        {"index_tensor": ParserIndex(features=np.asarray([0, 1]), global_shift=2)},
        {"index_tensor": ParserIndex(features=np.asarray([0, 2]), global_shift=3)},
    ]
    collate_fn = CollateAll(data_types={"index_tensor": "tensor"})

    result = collate_fn(batch)
    assert isinstance(result["index_tensor"], IndexBatch)


def test_collate_edge_index_tensor_returns_edge_index_batch():
    """Two-dimensional index tensors should produce an EdgeIndexBatch."""
    batch = [
        {
            "edge_tensor": ParserEdgeIndex(
                features=np.asarray([[0, 1], [1, 0]]), global_shift=2
            )
        },
        {
            "edge_tensor": ParserEdgeIndex(
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
            "index_tensor": ParserIndexList(
                features=[np.asarray([0, 2]), np.asarray([1])],
                global_shift=3,
                single_counts=np.asarray([2, 1]),
            )
        },
        {
            "index_tensor": ParserIndexList(
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


def test_collate_split_coordinate_tensor_empty_modules(monkeypatch):
    """Split coordinate collation should preserve zero-count modules."""

    class DummyTPC:
        num_modules = 2

    class DummyGeo:
        tpc = DummyTPC()

        @staticmethod
        def split(coords, target_id, meta=None):
            return coords, [
                np.asarray([0], dtype=np.int64),
                np.asarray([], dtype=np.int64),
            ]

    monkeypatch.setattr("spine.io.collate.GeoManager.get_instance", lambda: DummyGeo())

    tensor = ParserTensor(
        coords=np.asarray([[0, 0, 0]], dtype=np.float32),
        features=np.asarray([[1.0]], dtype=np.float32),
        meta=make_meta(),
    )
    result = CollateAll(data_types={"voxels": "tensor"}, split=True)(
        [{"voxels": tensor}]
    )

    assert result["voxels"].counts.tolist() == [1, 0]


def test_collate_split_feature_tensors_without_source_mapping(monkeypatch):
    """Split feature collation should fall back to plain concatenation without sources."""

    class DummyTPC:
        num_modules = 2

    class DummyGeo:
        tpc = DummyTPC()

    monkeypatch.setattr("spine.io.collate.GeoManager.get_instance", lambda: DummyGeo())

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
    result = CollateAll(data_types={"feat": "tensor"}, split=True)(batch)

    assert isinstance(result["feat"], TensorBatch)
    assert result["feat"].counts.tolist() == [2, 1]


def test_collate_index_list_without_single_counts():
    """Index-list collation should infer single counts when they are absent."""
    batch = [
        {
            "index_tensor": ParserIndexList(
                features=[np.asarray([0, 2]), np.asarray([1])],
                global_shift=3,
            )
        },
        {
            "index_tensor": ParserIndexList(
                features=[np.asarray([0, 1, 2])],
                global_shift=3,
            )
        },
    ]
    result = CollateAll(data_types={"index_tensor": "tensor"})(batch)

    assert isinstance(result["index_tensor"], IndexBatch)
    assert result["index_tensor"].counts.tolist() == [2, 1]
    assert result["index_tensor"].single_counts.tolist() == [2, 1, 3]
