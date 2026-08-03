"""Tests for compact structured cluster labels."""

import numpy as np
import pytest

from spine.data import ClusterLabelBatch
from spine.data import ClusterLabelBatch as BatchClusterLabel
from spine.data import ClusterLabelData, TensorBatch


def test_cluster_label_batch_is_exported_from_batch_namespace():
    """The batched cluster-label container should live under data.batch."""
    assert ClusterLabelBatch is BatchClusterLabel


def particle_fields(array=np.asarray):
    """Build a two-particle table with one ancestor relationship."""
    return {
        "particle": array([10, 11]),
        "group": array([10, 10]),
        "ancestor": array([0, 0]),
        "interaction": array([2, 2]),
        "nu": array([0, 0]),
        "pid": array([2, 3]),
        "group_primary": array([1, 0]),
        "interaction_primary": array([1, 0]),
        "vertex": array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "momentum": array([200.0, 300.0]),
        "energy_init": array([220.0, 330.0]),
        "shape": array([1, 0]),
    }


def test_cluster_label_expands_named_particle_fields():
    """Named and derived fields should expand through voxel associations."""
    data = np.asarray(
        [
            [0.0, 1.0, 2.0, 4.0, 7.0, 0.0],
            [1.0, 1.0, 2.0, 5.0, 8.0, 1.0],
            [2.0, 1.0, 2.0, 6.0, -1.0, -1.0],
        ]
    )
    labels = ClusterLabelData(data, particle_fields())

    np.testing.assert_array_equal(labels.voxel_field("cluster"), [7, 8, -1])
    np.testing.assert_array_equal(labels.voxel_field("particle"), [10, 11, -1])
    np.testing.assert_array_equal(labels.voxel_field("pid"), [2, 3, -1])
    np.testing.assert_array_equal(labels.voxel_field("ancestor_pid"), [2, 2, -1])
    np.testing.assert_allclose(
        labels.voxel_field("ancestor_momentum"), [200.0, 200.0, -1.0]
    )
    np.testing.assert_array_equal(labels.voxel_field("vertex_x"), [1.0, 1.0, -1])


def test_cluster_label_exposes_fixed_voxel_field_aliases():
    """Stable voxel fields should be available through explicit properties."""
    data = np.asarray(
        [
            [0.0, 1.0, 2.0, 4.0, 7.0, 0.0],
            [1.0, 1.0, 2.0, 5.0, 8.0, 1.0],
        ]
    )
    labels = ClusterLabelData(data, particle_fields())
    aliases = {
        "values": "value",
        "cluster_ids": "cluster",
        "particle_indexes": "particle_index",
        "particle_ids": "particle",
        "group_ids": "group",
        "ancestor_indexes": "ancestor",
        "interaction_ids": "interaction",
        "neutrino_ids": "nu",
        "pids": "pid",
        "group_primaries": "group_primary",
        "interaction_primaries": "interaction_primary",
        "vertices": "vertex",
        "momenta": "momentum",
        "energies": "energy_init",
        "shapes": "shape",
        "ancestor_pids": "ancestor_pid",
        "ancestor_momenta": "ancestor_momentum",
    }

    for alias, field in aliases.items():
        np.testing.assert_array_equal(
            getattr(labels, alias),
            labels.voxel_field(field),
        )


def test_cluster_label_without_particle_information_is_explicit():
    """Association-only labels should expose cluster/value but no particles."""
    labels = ClusterLabelData(np.zeros((2, 5), dtype=np.float32))

    assert labels.voxel_field("value").shape == (2,)
    assert labels.voxel_field("cluster").shape == (2,)
    with pytest.raises(ValueError, match="particle information"):
        labels.voxel_field("pid")


def test_cluster_label_batch_round_trip_and_selection():
    """Device conversion, event slicing, and row selection preserve tables."""
    torch = pytest.importorskip("torch")
    data = TensorBatch(
        np.asarray(
            [
                [0, 0, 0, 0, 1, 4, 0],
                [0, 1, 0, 0, 2, 5, 1],
                [1, 0, 1, 0, 3, 6, 0],
            ],
            dtype=np.float32,
        ),
        counts=np.asarray([2, 1]),
        has_batch_col=True,
    )
    particles = {
        name: TensorBatch(np.concatenate((value, value[:1])), counts=[2, 1])
        for name, value in particle_fields().items()
    }
    labels = ClusterLabelBatch(data, particles)

    tensor_labels = labels.to_tensor(device="cpu")
    assert isinstance(tensor_labels.tensor, torch.Tensor)
    np.testing.assert_array_equal(
        tensor_labels.to_numpy().voxel_field("pid").data,
        np.asarray([2, 3, 2]),
    )
    np.testing.assert_array_equal(labels[1].voxel_field("particle"), [10])

    selected = labels.select(np.asarray([1, 2]), counts=np.asarray([1, 1]))
    np.testing.assert_array_equal(selected.voxel_field("pid").data, [3, 2])
    np.testing.assert_array_equal(selected.pids.data, [3, 2])
    assert selected.particles is labels.particles


def test_cluster_label_rejects_invalid_particle_association():
    """Voxel associations may not point outside the event particle table."""
    data = np.asarray([[0, 0, 0, 1, 0, 2]], dtype=np.float32)
    with pytest.raises(ValueError, match="outside the particle table"):
        ClusterLabelData(data, particle_fields())
