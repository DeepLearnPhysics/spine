"""Test that the collate function(s) work as intended."""

import numpy as np
import pytest

from spine.data import (
    ClusterLabelBatch,
    ClusterLabelData,
    EdgeIndexBatch,
    EdgeIndexData,
    IndexBatch,
    IndexData,
    IndexListData,
    Meta,
    ObjectListBatch,
    ObjectListData,
    TensorBatch,
    TensorData,
    TensorSchema,
)
from spine.geo import GeoManager
from spine.io.collate import CollateAll


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

            data[f"sparse_{name}"] = TensorData(
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

            data[f"edge_index_{name}"] = EdgeIndexData(features=edge_index, span=10)

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
        data_keys={key: "tensor" for key in batch_sparse[0].keys()}, split=split
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
        data_keys={key: "tensor" for key in batch_edge_index[0].keys()}
    )

    # Pass the batch through the collate function
    result = collate_fn(batch_edge_index)

    # Check that each key in the output if of the same length as the batch
    for k in batch_edge_index[0]:
        assert len(result[k]) == len(batch_edge_index)


def test_collate_cluster_labels_preserves_event_local_particle_tables():
    """Cluster-label collation should keep associations local to each event."""
    particle_a = {"particle": np.asarray([10]), "pid": np.asarray([2])}
    particle_b = {
        "particle": np.asarray([20, 21]),
        "pid": np.asarray([3, 4]),
    }
    batch = [
        {
            "label": ClusterLabelData(
                coords=np.asarray([[1, 2, 3]], dtype=np.int32),
                features=np.asarray([[4.0, 7.0, 0.0]], dtype=np.float32),
                particles=particle_a,
            )
        },
        {
            "label": ClusterLabelData(
                coords=np.asarray([[4, 5, 6], [7, 8, 9]], dtype=np.int32),
                features=np.asarray(
                    [[5.0, 8.0, 0.0], [6.0, 9.0, 1.0]], dtype=np.float32
                ),
                particles=particle_b,
            )
        },
    ]

    result = CollateAll(data_keys={"label": "cluster_label"})(batch)["label"]

    assert isinstance(result, ClusterLabelBatch)
    assert result.counts.tolist() == [1, 2]
    assert result.particle_field("pid").counts.tolist() == [1, 2]
    np.testing.assert_array_equal(result.voxel_field("pid").data, [2, 3, 4])
    np.testing.assert_array_equal(result[1].voxel_field("particle"), [20, 21])


def test_collate_requires_data_keys():
    """Collation should reject an unspecified dataset product contract."""
    with pytest.raises(ValueError, match="data_keys"):
        CollateAll(data_keys=None)


def test_collate_object_list_products():
    """Typed event object lists should retain their batch boundary."""
    batch = [
        {"objects": ObjectListData([1], 0)},
        {"objects": ObjectListData([2, 3], 0)},
    ]

    result = CollateAll(data_keys=("objects",))(batch)["objects"]

    assert isinstance(result, ObjectListBatch)
    assert [list(entry) for entry in result] == [[1], [2, 3]]


def test_collate_cluster_label_validation():
    """Cluster-label collation should enforce event and particle schemas."""
    label = ClusterLabelData(
        coords=np.asarray([[0, 0, 0]]),
        features=np.asarray([[1.0, 0.0, 0.0]]),
        particles={"particle": np.asarray([0]), "pid": np.asarray([1])},
    )
    no_particles = ClusterLabelData(
        coords=np.asarray([[1, 1, 1]]),
        features=np.asarray([[1.0, 0.0]]),
    )
    other_fields = ClusterLabelData(
        coords=np.asarray([[1, 1, 1]]),
        features=np.asarray([[1.0, 0.0, 0.0]]),
        particles={"particle": np.asarray([0]), "energy": np.asarray([2.0])},
    )
    collate = CollateAll(data_keys=("label",))

    with pytest.raises(TypeError, match="ClusterLabelData"):
        collate.stack_cluster_labels([{"label": label}, {"label": object()}], "label")

    with pytest.raises(ValueError, match="Particle information"):
        collate.stack_cluster_labels(
            [{"label": label}, {"label": no_particles}], "label"
        )

    with pytest.raises(ValueError, match="fields must be consistent"):
        collate.stack_cluster_labels(
            [{"label": label}, {"label": other_fields}], "label"
        )


def test_collate_rejects_mismatched_tensor_schemas():
    """Coordinate and feature-only products must agree on their schemas."""
    coords = np.asarray([[0, 0, 0]])
    features = np.asarray([[1.0]])
    coord_batch = [
        {
            "value": TensorData(
                coords=coords,
                features=features,
                schema=TensorSchema(feature_fields={"value": (0,)}),
            )
        },
        {
            "value": TensorData(
                coords=coords,
                features=features,
                schema=TensorSchema(feature_fields={"other": (0,)}),
            )
        },
    ]
    collate = CollateAll(data_keys=("value",))
    with pytest.raises(ValueError, match="schemas do not match"):
        collate.stack_coord_tensors(coord_batch, "value")

    feature_batch = [
        {
            "value": TensorData(
                features=features,
                feats_only=True,
                schema=TensorSchema(feature_fields={"value": (0,)}, feats_only=True),
            )
        },
        {
            "value": TensorData(
                features=features,
                feats_only=True,
                schema=TensorSchema(feature_fields={"other": (0,)}, feats_only=True),
            )
        },
    ]
    with pytest.raises(ValueError, match="schemas do not match"):
        collate.stack_feat_tensors(feature_batch, "value")


def test_collate_scalar():
    """Tests the collation of scalar values."""
    # Initialize the collation class
    collate_fn = CollateAll(data_keys={"scalar": "scalar"})

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
    collate_fn = CollateAll(data_keys={"list": "list"})

    # Initialize a simple batch of lists
    batch_list = [{"list": [i] * i} for i in range(4)]

    # Pass the batch through the collate function
    result = collate_fn(batch_list)

    # Check that each key in the output if of the same length as the batch
    assert len(result["list"]) == len(batch_list)

    # Check that the input is intact
    for i, data in enumerate(batch_list):
        assert data["list"] == result["list"][i]


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
            "voxels": TensorData(
                coords=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
                features=np.asarray([[1.0], [2.0]], dtype=np.float32),
                meta=meta,
            )
        },
        {
            "voxels": TensorData(
                coords=np.asarray([[2, 2, 2]], dtype=np.int64),
                features=np.asarray([[3.0]], dtype=np.float32),
                meta=meta,
            )
        },
    ]
    result = CollateAll(data_keys={"voxels": "tensor"})(batch)

    tensor = result["voxels"]
    assert isinstance(tensor, TensorBatch)
    assert tensor.counts.tolist() == [2, 1]
    assert tensor.tensor.shape == (3, 5)


def test_collate_index_tensor_and_edge_tensor_offsets():
    """Index-like tensors should be offset and wrapped in the right batch type."""
    collate_fn = CollateAll(data_keys={"flat": "tensor", "edge": "tensor"})
    batch = [
        {
            "flat": IndexData(features=np.asarray([0, 1], dtype=np.int64), span=2),
            "edge": EdgeIndexData(
                features=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
                span=2,
            ),
        },
        {
            "flat": IndexData(features=np.asarray([0], dtype=np.int64), span=1),
            "edge": EdgeIndexData(
                features=np.asarray([[0], [0]], dtype=np.int64),
                span=1,
            ),
        },
    ]
    result = collate_fn(batch)

    assert isinstance(result["flat"], IndexBatch)
    assert result["flat"].index.tolist() == [0, 1, 2]
    assert result["flat"].spans.tolist() == [2, 1]
    assert isinstance(result["edge"], EdgeIndexBatch)
    assert result["edge"].index.tolist() == [[0, 1, 2], [1, 0, 2]]
    assert result["edge"].spans.tolist() == [2, 1]


def test_collate_with_overlay():
    """CollateAll should apply overlay before batching."""
    collate_fn = CollateAll(
        data_keys={"run": "scalar"},
        overlay={"multiplicity": 2},
        overlay_methods={"run": "match"},
    )

    result = collate_fn([{"run": 1}, {"run": 1}])
    assert result == {"run": [1]}


def test_collate_overlay_requires_overlay_methods():
    """CollateAll should require overlay methods when overlaying is enabled."""
    with pytest.raises(ValueError, match="overlay_methods"):
        CollateAll(data_keys={"run": "scalar"}, overlay={"multiplicity": 2})


def test_collate_index_tensor_returns_index_batch():
    """One-dimensional index tensors should produce an IndexBatch."""
    batch = [
        {"index_tensor": IndexData(features=np.asarray([0, 1]), span=2)},
        {"index_tensor": IndexData(features=np.asarray([0, 2]), span=3)},
    ]
    collate_fn = CollateAll(data_keys={"index_tensor": "tensor"})

    result = collate_fn(batch)
    assert isinstance(result["index_tensor"], IndexBatch)


def test_collate_edge_index_tensor_returns_edge_index_batch():
    """Two-dimensional index tensors should produce an EdgeIndexBatch."""
    batch = [
        {"edge_tensor": EdgeIndexData(features=np.asarray([[0, 1], [1, 0]]), span=2)},
        {"edge_tensor": EdgeIndexData(features=np.asarray([[0, 1], [1, 0]]), span=2)},
    ]
    collate_fn = CollateAll(data_keys={"edge_tensor": "tensor"})

    result = collate_fn(batch)
    assert isinstance(result["edge_tensor"], EdgeIndexBatch)


def test_collate_index_list_tensor_returns_index_batch():
    """List-backed index tensors should produce an IndexBatch with per-index sizes."""
    batch = [
        {
            "index_tensor": IndexListData(
                features=[np.asarray([0, 2]), np.asarray([1])],
                span=3,
                single_counts=np.asarray([2, 1]),
            )
        },
        {
            "index_tensor": IndexListData(
                features=[np.asarray([0, 1, 2])],
                span=3,
            )
        },
    ]
    collate_fn = CollateAll(data_keys={"index_tensor": "tensor"})

    result = collate_fn(batch)
    assert isinstance(result["index_tensor"], IndexBatch)
    assert result["index_tensor"].counts.tolist() == [2, 1]
    assert result["index_tensor"].single_counts.tolist() == [2, 1, 3]
    assert result["index_tensor"].spans.tolist() == [3, 3]


def test_collate_feature_tensors_without_coords():
    """Feature-only tensors should be collated with stack_feat_tensors."""
    batch = [
        {
            "feat": TensorData(
                features=np.asarray([[1.0], [2.0]], dtype=np.float32), feats_only=True
            )
        },
        {
            "feat": TensorData(
                features=np.asarray([[3.0]], dtype=np.float32), feats_only=True
            )
        },
    ]
    collate_fn = CollateAll(data_keys={"feat": "tensor"})

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
            "feat": TensorData(
                features=np.asarray([[10.0], [20.0]], dtype=np.float32), feats_only=True
            ),
            "source": TensorData(
                features=np.asarray([[0], [1]], dtype=np.int64), feats_only=True
            ),
        }
    ]
    collate_fn = CollateAll(
        data_keys={"feat": "tensor"},
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

    tensor = TensorData(
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
    collate_fn = CollateAll(data_keys={"voxels": "tensor"}, split=True)

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

    tensor = TensorData(
        coords=np.asarray([[0, 0, 0]], dtype=np.float32),
        features=np.asarray([[1.0]], dtype=np.float32),
        meta=make_meta(),
    )
    result = CollateAll(data_keys={"voxels": "tensor"}, split=True)(
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
            "feat": TensorData(
                features=np.asarray([[1.0], [2.0]], dtype=np.float32), feats_only=True
            )
        },
        {
            "feat": TensorData(
                features=np.asarray([[3.0]], dtype=np.float32), feats_only=True
            )
        },
    ]
    result = CollateAll(data_keys={"feat": "tensor"}, split=True)(batch)

    assert isinstance(result["feat"], TensorBatch)
    assert result["feat"].counts.tolist() == [2, 1]


def test_collate_index_list_without_single_counts():
    """Index-list collation should infer single counts when they are absent."""
    batch = [
        {
            "index_tensor": IndexListData(
                features=[np.asarray([0, 2]), np.asarray([1])],
                span=3,
            )
        },
        {
            "index_tensor": IndexListData(
                features=[np.asarray([0, 1, 2])],
                span=3,
            )
        },
    ]
    result = CollateAll(data_keys={"index_tensor": "tensor"})(batch)

    assert isinstance(result["index_tensor"], IndexBatch)
    assert result["index_tensor"].counts.tolist() == [2, 1]
    assert result["index_tensor"].single_counts.tolist() == [2, 1, 3]
