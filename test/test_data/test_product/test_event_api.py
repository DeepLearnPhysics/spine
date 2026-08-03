"""Behavioral coverage for event-level self-describing products."""

import numpy as np
import pytest

from spine.data import (
    ClusterLabelData,
    EdgeIndexData,
    IndexData,
    IndexListData,
    TensorData,
    TensorSchema,
)


def _particle_fields(array=np.asarray):
    """Return a complete two-particle truth table."""
    return {
        "particle": array([10, 11]),
        "group": array([4, 4]),
        "ancestor": array([0, -1]),
        "interaction": array([2, 2]),
        "nu": array([0, 0]),
        "pid": array([2, 3]),
        "group_primary": array([1, 0]),
        "interaction_primary": array([1, 0]),
        "vertex": array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "momentum": array([200.0, 300.0]),
        "energy_init": array([220.0, 330.0]),
        "shape": array([1, 0]),
    }


def test_tensor_data_schema_metadata_and_named_access():
    """Tensor products should expose every logical schema component."""
    coords = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    features = np.asarray([[7, 8], [9, 10]], dtype=np.float32)
    tensor = TensorData(
        features,
        coords,
        coordinate_groups={"point": (0, 1, 2)},
        feature_fields={"score": (0,), "shape": (1,)},
        index_cols=np.asarray([1]),
        remove_duplicates=True,
        sum_cols=np.asarray([0]),
        avg_cols=np.asarray([1]),
        prec_col=1,
        precedence=np.asarray([2, 1, 0]),
        overlay_reference="data",
    )

    assert len(tensor) == 2
    assert tensor.shape == (2, 5)
    assert tensor.coordinate_groups == {"point": (0, 1, 2)}
    np.testing.assert_array_equal(tensor.coords, coords)
    np.testing.assert_array_equal(tensor.coordinates("point"), coords)
    np.testing.assert_array_equal(tensor.feature("shape"), [[8], [10]])
    np.testing.assert_array_equal(tensor[0], [1, 2, 3, 7, 8])
    np.testing.assert_array_equal(tensor.index_cols, [1])
    np.testing.assert_array_equal(tensor.sum_cols, [0])
    np.testing.assert_array_equal(tensor.avg_cols, [1])
    np.testing.assert_array_equal(tensor.precedence, [2, 1, 0])
    assert tensor.remove_duplicates is True
    assert tensor.prec_col == 1
    assert tensor.feats_only is False
    assert tensor.overlay_reference == "data"

    metadata = TensorData.metadata(tensor.schema)
    assert metadata["product_type"] == "tensor"
    assert TensorSchema.from_dict(metadata["schema"]) == tensor.schema


def test_tensor_data_mutation_numpy_protocol_and_errors():
    """Compatibility access should remain explicit and fail on ambiguity."""
    tensor = TensorData(np.asarray([[1.0], [2.0]]), feats_only=True)
    assert tensor.coords is None
    np.testing.assert_array_equal(tensor.values, [1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(tensor, dtype=np.float64), [[1.0], [2.0]])
    copied = np.array(tensor, copy=True)
    assert copied is not tensor.features

    coords = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    tensor.coordinate_data = coords
    np.testing.assert_array_equal(tensor.coordinate_data, coords)
    tensor.coords = coords + 1
    np.testing.assert_array_equal(tensor.coordinate_data, coords + 1)

    with pytest.raises(KeyError, match="Unknown coordinate group"):
        tensor.coordinates("missing")
    with pytest.raises(KeyError, match="Unknown feature field"):
        tensor.feature("missing")
    with pytest.raises(ValueError, match="exactly one feature column"):
        _ = TensorData(np.ones((2, 2))).values
    with pytest.raises(ValueError, match="no coordinates"):
        TensorData(np.ones((2, 1)), feats_only=True).coordinates()


def test_tensor_data_rejects_mixed_schema_configuration():
    """A serialized schema must not be partially overridden."""
    with pytest.raises(ValueError, match="Do not combine"):
        TensorData(
            np.ones((1, 1)),
            schema=TensorSchema(feats_only=True),
            feature_fields={"value": (0,)},
        )


def test_index_and_edge_event_products_behave_like_arrays():
    """Index products should retain metadata while supporting array consumers."""
    index = IndexData(np.asarray([1, 3, 5]), span=7)
    assert len(index) == 3
    assert index[1] == 3
    converted = np.asarray(index, dtype=np.float32)
    np.testing.assert_array_equal(converted, [1.0, 3.0, 5.0])
    assert np.array(index, copy=True) is not index.features

    members = [np.asarray([0, 2]), np.asarray([1])]
    listed = IndexListData(members, span=3, single_counts=np.asarray([2, 1]))
    assert len(listed) == 2
    assert list(listed) == members
    np.testing.assert_array_equal(listed[1], [1])

    edge = EdgeIndexData(np.asarray([[0, 1], [1, 2]]), span=3, directed=False)
    assert len(edge) == 2
    assert edge.index is edge.features
    np.testing.assert_array_equal(edge.index_t, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(edge[:, 0], [0, 1])
    np.testing.assert_array_equal(np.asarray(edge, dtype=np.float32), edge.features)
    assert np.array(edge, copy=True) is not edge.features


def test_cluster_label_construction_and_validation():
    """Packed and split labels should enforce their compact association schema."""
    coords = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    features = np.asarray([[4, 7, 0], [5, 8, -1]], dtype=np.float32)
    labels = ClusterLabelData(
        coords=coords,
        features=features,
        particles=_particle_fields(),
        meta="meta",
        sum_cols=np.asarray([0]),
    )

    assert len(labels) == 2
    assert labels.meta == "meta"
    np.testing.assert_array_equal(labels.coords, coords)
    np.testing.assert_array_equal(labels.features, features)
    assert labels.metadata()["feature_fields"]["particle_index"] == (2,)
    assert "particle_index" not in labels.metadata(False)["feature_fields"]

    with pytest.raises(ValueError, match="packed `data`"):
        ClusterLabelData(labels.data, coords=coords, features=features)
    with pytest.raises(ValueError, match="either `data`"):
        ClusterLabelData()
    with pytest.raises(ValueError, match="must have columns"):
        ClusterLabelData(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="must be omitted"):
        ClusterLabelData(np.zeros((1, 6)))
    with pytest.raises(ValueError, match="requires a particle-index"):
        ClusterLabelData(np.zeros((1, 5)), _particle_fields())

    inconsistent = _particle_fields()
    inconsistent["pid"] = np.asarray([2])
    with pytest.raises(ValueError, match="same length"):
        ClusterLabelData(np.zeros((1, 6)), inconsistent)


def test_cluster_label_particle_and_voxel_field_errors():
    """Particle aliases, virtual fields and absent information should be clear."""
    data = np.asarray(
        [[0, 1, 2, 4, 7, 0], [3, 4, 5, 5, 8, 1], [6, 7, 8, 6, -1, -1]],
        dtype=np.float32,
    )
    labels = ClusterLabelData(data, _particle_fields())

    np.testing.assert_array_equal(labels.particle_field("part"), [10, 11])
    np.testing.assert_array_equal(labels.particle_field("type"), [2, 3])
    np.testing.assert_array_equal(labels.particle_field("vertex_y"), [2, 5])
    np.testing.assert_array_equal(labels.particle_field("ancestor_pid"), [2, -1])
    np.testing.assert_array_equal(labels.voxel_field("particle_index"), [0, 1, -1])
    np.testing.assert_array_equal(labels.voxel_field("value"), [4, 5, 6])
    np.testing.assert_array_equal(labels.voxel_field("cluster"), [7, 8, -1])

    with pytest.raises(KeyError, match="unavailable"):
        labels.particle_field("missing")

    compact = ClusterLabelData(data[:, :5])
    with pytest.raises(ValueError, match="particle information"):
        compact.particle_field("pid")
    with pytest.raises(ValueError, match="Particle information"):
        compact._ancestor_field("pid")
    with pytest.raises(ValueError, match="Particle indexes"):
        compact.voxel_field("particle_index")


def test_torch_backed_event_product_paths():
    """Event products should preserve PyTorch storage until NumPy is requested."""
    torch = pytest.importorskip("torch")
    coords = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    features = torch.tensor([[6.0], [7.0]])
    tensor = TensorData(features, coords)
    assert torch.equal(tensor.data, torch.cat((coords, features), dim=1))
    np.testing.assert_array_equal(np.asarray(tensor, dtype=np.float64), tensor.data)

    index = IndexData(torch.tensor([1, 2]), span=3)
    edge = EdgeIndexData(torch.tensor([[0, 1], [1, 0]]), span=2)
    np.testing.assert_array_equal(np.asarray(index, dtype=np.float32), [1, 2])
    np.testing.assert_array_equal(np.asarray(edge, dtype=np.float32), edge.features)

    particles = _particle_fields(torch.tensor)
    labels = ClusterLabelData(
        torch.tensor([[0, 1, 2, 4, 7, 0], [3, 4, 5, 5, 8, 1]]),
        particles,
    )
    assert torch.equal(labels.particle_field("ancestor_pid"), torch.tensor([2, -1]))
    assert torch.equal(labels.voxel_field("pid"), torch.tensor([2, 3]))

    split_labels = ClusterLabelData(
        coords=torch.tensor([[0, 1, 2]]),
        features=torch.tensor([[4, 7, 0]]),
        particles=particles,
    )
    assert torch.equal(split_labels.data, torch.tensor([[0, 1, 2, 4, 7, 0]]))
