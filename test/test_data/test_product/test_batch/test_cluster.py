"""Behavioral coverage for batched cluster-label products."""

import numpy as np
import pytest

from spine.data import (
    ClusterLabelBatch,
    ObjectListBatch,
    ObjectListData,
    TensorBatch,
)


def _cluster_batch(with_particles=True, meta=None):
    """Build a two-event cluster-label batch with local associations."""
    rows = np.asarray(
        [
            [0, 0, 0, 0, 1, 7, 0],
            [0, 1, 0, 0, 2, 8, 1],
            [1, 0, 1, 0, 3, 9, 0],
            [1, 1, 1, 0, 4, -1, -1],
        ],
        dtype=np.float32,
    )
    if not with_particles:
        rows = rows[:, :-1]
    data = TensorBatch(
        rows,
        counts=[2, 2],
        has_batch_col=True,
        meta=meta,
    )
    particles = None
    if with_particles:
        counts = [2, 1]
        particles = {
            "particle": TensorBatch(np.asarray([10, 11, 20]), counts),
            "ancestor": TensorBatch(np.asarray([0, -1, 0]), counts),
            "pid": TensorBatch(np.asarray([2, 3, 4]), counts),
            "momentum": TensorBatch(np.asarray([100.0, 200.0, 300.0]), counts),
            "vertex": TensorBatch(
                np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
                counts,
            ),
        }
    return ClusterLabelBatch(data, particles)


def test_cluster_batch_normalizes_and_exposes_compact_contract():
    """Construction should normalize coordinates, schema and metadata."""
    labels = _cluster_batch(meta=["a", "b"])

    assert len(labels) == 2
    assert labels.batch_size == 2
    assert labels.is_numpy is True
    assert labels.device is None
    assert labels.dtype == np.float32
    assert labels.meta == ["a", "b"]
    assert labels.tensor is labels.numpy_tensor()
    assert labels.to_tensor_batch() is labels.data
    np.testing.assert_array_equal(labels.counts, [2, 2])
    np.testing.assert_array_equal(labels.data.coord_cols, [1, 2, 3])
    np.testing.assert_array_equal(labels.coords.data, labels.tensor[:, 1:4])
    np.testing.assert_array_equal(labels.coordinates("points").data, labels.coords.data)
    np.testing.assert_array_equal(labels.coordinate_data.data, labels.coords.data)

    event = labels[1]
    assert event.meta == "b"
    np.testing.assert_array_equal(event.coords, [[0, 1, 0], [1, 1, 0]])
    np.testing.assert_array_equal(event.particle_field("particle"), [20])
    np.testing.assert_array_equal(event.voxel_field("pid"), [4, -1])


def test_cluster_batch_named_particle_and_voxel_fields():
    """Virtual fields should resolve through each event's local particle table."""
    labels = _cluster_batch()

    np.testing.assert_array_equal(labels.particle_field("part").data, [10, 11, 20])
    np.testing.assert_array_equal(labels.particle_field("vertex_z").data, [3, 6, 9])
    np.testing.assert_array_equal(
        labels.particle_field("ancestor_pid").data, [2, -1, 4]
    )
    np.testing.assert_array_equal(
        labels.particle_field("ancestor_momentum").data, [100, -1, 300]
    )
    np.testing.assert_array_equal(labels.voxel_field("value").data, [1, 2, 3, 4])
    np.testing.assert_array_equal(labels.voxel_field("cluster").data, [7, 8, 9, -1])
    np.testing.assert_array_equal(
        labels.voxel_field("particle_index").data, [0, 1, 0, -1]
    )
    np.testing.assert_array_equal(labels.voxel_field("pid").data, [2, 3, 4, -1])

    with pytest.raises(KeyError, match="unavailable"):
        labels.particle_field("missing")


def test_cluster_batch_selection_preserves_particle_tables():
    """Voxel selection should retain event-local particle associations."""
    labels = _cluster_batch(meta=["a", "b"])
    selected = labels.select(np.asarray([1, 2]), counts=np.asarray([1, 1]))

    assert selected.particles is labels.particles
    assert selected.meta == labels.meta
    np.testing.assert_array_equal(selected.voxel_field("pid").data, [3, 4])
    np.testing.assert_array_equal(selected.counts, [1, 1])


def test_cluster_batch_without_particle_information_is_explicit():
    """Association-free batches should retain compact voxel functionality."""
    labels = _cluster_batch(with_particles=False)

    np.testing.assert_array_equal(labels.values.data, [1, 2, 3, 4])
    np.testing.assert_array_equal(labels.cluster_ids.data, [7, 8, 9, -1])
    assert labels.to_numpy() is not None
    with pytest.raises(ValueError, match="particle information"):
        labels.particle_field("pid")
    with pytest.raises(ValueError, match="Particle indexes"):
        labels.voxel_field("particle_index")
    with pytest.raises(TypeError, match="not backed by a torch.Tensor"):
        labels.torch_tensor()


def test_cluster_batch_rejects_invalid_layout_and_metadata():
    """Malformed compact batches should fail at their construction boundary."""
    rows = np.zeros((2, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="batch column"):
        ClusterLabelBatch(TensorBatch(rows, counts=[2]))

    with pytest.raises(ValueError, match="must have 6 columns"):
        ClusterLabelBatch(
            TensorBatch(
                np.zeros((1, 5)), counts=[1], has_batch_col=True, coord_cols=[1, 2, 3]
            )
        )

    with pytest.raises(ValueError, match="columns 1, 2 and 3"):
        ClusterLabelBatch(
            TensorBatch(
                np.zeros((1, 6)), counts=[1], has_batch_col=True, coord_cols=[2, 3, 4]
            )
        )

    with pytest.raises(ValueError, match="one item per event"):
        ClusterLabelBatch(
            TensorBatch(np.zeros((1, 6)), counts=[1], has_batch_col=True),
            meta=[],
        )


def test_cluster_batch_rejects_invalid_particle_tables():
    """Particle fields must align with each other and their voxel associations."""
    data = TensorBatch(
        np.asarray([[0, 0, 0, 0, 1, 2, 1]], dtype=np.float32),
        counts=[1],
        has_batch_col=True,
    )
    wrong_batch = {"pid": TensorBatch(np.asarray([1]), counts=[0, 1])}
    with pytest.raises(ValueError, match="wrong batch size"):
        ClusterLabelBatch(data, wrong_batch)

    bad_counts = {
        "pid": TensorBatch(np.asarray([1]), counts=[1]),
        "shape": TensorBatch(np.asarray([0, 1]), counts=[2]),
    }
    with pytest.raises(ValueError, match="share event counts"):
        ClusterLabelBatch(data, bad_counts)

    outside = {"pid": TensorBatch(np.asarray([1]), counts=[1])}
    with pytest.raises(ValueError, match="outside its event table"):
        ClusterLabelBatch(data, outside)


def test_object_list_batch_contract():
    """Object batches should validate entries and retain list behavior."""
    entries = [ObjectListData([1], 0), ObjectListData([], 0)]
    batch = ObjectListBatch(iter(entries))

    assert batch.batch_size == 2
    assert batch.event(0) is entries[0]
    with pytest.raises(TypeError, match="ObjectListData"):
        ObjectListBatch([entries[0], []])


def test_torch_cluster_batch_conversion_and_backend_validation():
    """Conversion should keep voxel and particle tables on one backend."""
    torch = pytest.importorskip("torch")
    labels = _cluster_batch(meta=["a", "b"])
    tensor_labels = labels.to_tensor(dtype=torch.float32, device="cpu")

    assert tensor_labels.is_numpy is False
    assert tensor_labels.device.type == "cpu"
    assert tensor_labels.torch_tensor() is tensor_labels.tensor
    assert torch.equal(
        tensor_labels.voxel_field("pid").data, torch.tensor([2, 3, 4, -1])
    )
    assert torch.equal(
        tensor_labels.particle_field("ancestor_pid").data,
        torch.tensor([2, -1, 4]),
    )
    restored = tensor_labels.to_numpy()
    np.testing.assert_array_equal(restored.voxel_field("pid").data, [2, 3, 4, -1])

    mixed = {"pid": TensorBatch(torch.tensor([1]), counts=[1])}
    data = TensorBatch(
        np.asarray([[0, 0, 0, 0, 1, 2, 0]], dtype=np.float32),
        counts=[1],
        has_batch_col=True,
    )
    with pytest.raises(ValueError, match="different array backend"):
        ClusterLabelBatch(data, mixed)
