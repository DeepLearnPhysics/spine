"""Regression tests for structured cluster-label adaptation."""

from inspect import signature

import numpy as np
import pytest

from spine.constants import DELTA_SHP, MICHL_SHP, SHOWR_SHP, TRACK_SHP
from spine.data import (
    ClusterLabelBatch,
    ClusterLabelData,
    IndexBatch,
    TensorBatch,
    TensorData,
)
from spine.model.full_chain.label import ClusterLabelAdapter
from spine.utils.conditional import torch


class PropagationAdapter(ClusterLabelAdapter):
    """Expose internal propagation solely for focused algorithm tests."""

    @staticmethod
    def propagate(query_coords, source_coords, source_labels, weighted=True):
        """Run the inherited propagation implementation."""
        return PropagationAdapter._propagate(
            query_coords,
            source_coords,
            source_labels,
            weighted,
        )

    def to_numpy(self, array):
        """Expose backend validation solely for focused contract tests."""
        return self._to_numpy(array)


def test_adapter_exposes_break_class_defaults():
    """The constructor should advertise its immutable breakup-class default."""
    default = signature(ClusterLabelAdapter).parameters["break_classes"].default

    assert default == (SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP)
    assert ClusterLabelAdapter().break_classes == default


def test_adapter_preserves_invalid_cluster_ids_across_events():
    """A global event offset must never turn invalid ``-1`` IDs positive."""
    compact = TensorBatch(
        np.asarray(
            [
                [0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
        counts=[1, 1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    cluster_label = ClusterLabelBatch(compact)
    semantic_label = TensorBatch.from_data_list(
        [
            TensorData(np.asarray([0], dtype=np.float32), coords=np.zeros((1, 3))),
            TensorData(np.asarray([0], dtype=np.float32), coords=np.zeros((1, 3))),
        ]
    )
    semantic_prediction = TensorBatch(np.asarray([0, 5], dtype=np.int64), counts=[1, 1])

    adapted = ClusterLabelAdapter()(cluster_label, semantic_label, semantic_prediction)

    assert adapted.cluster_ids.data[0] >= 0
    assert adapted.cluster_ids.data[1] == -1
    assert adapted.data.schema == cluster_label.data.schema


def test_adapter_validates_aligned_torch_products():
    """The public adapter should enforce alignment on Torch-backed products."""
    compact = TensorBatch(
        torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.float32),
        counts=[1],
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )
    cluster_label = ClusterLabelBatch(compact)
    semantic_label = TensorBatch.from_data_list(
        [
            TensorData(
                torch.tensor([0.0, 5.0, 5.0]),
                coords=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
                ),
            )
        ]
    )
    semantic_prediction = TensorBatch(torch.tensor([0, 0, 0]), counts=[3])

    adapted = ClusterLabelAdapter()(
        cluster_label,
        semantic_label,
        semantic_prediction,
    )

    assert isinstance(adapted.tensor, torch.Tensor)
    assert torch.all(adapted.cluster_ids.data >= 0)


def test_adapter_rejects_mixed_backend_conversion():
    """CPU conversion should reject arrays from the inactive backend."""
    adapter = PropagationAdapter()
    with pytest.raises(TypeError, match="Expected a NumPy array"):
        adapter.to_numpy(torch.zeros(1))

    adapter.torch = True
    with pytest.raises(TypeError, match="Expected a Torch tensor"):
        adapter.to_numpy(np.zeros(1))


def adapter_for_numpy():
    """Return an adapter initialized for direct NumPy event processing."""
    return ClusterLabelAdapter()


def adapt_event(adapter, clusters, semantic, prediction, orig_index=None):
    """Run the public batch adapter interface for one NumPy event."""
    cluster_data = TensorData(
        clusters.features,
        coords=clusters.coords,
        schema=ClusterLabelData.tensor_schema(clusters.particles is not None),
    )
    cluster_batch = ClusterLabelBatch(TensorBatch.from_data_list([cluster_data]))
    semantic_batch = TensorBatch.from_data_list([semantic])
    prediction_batch = TensorBatch(prediction, counts=[len(prediction)])

    index_batch = None
    if orig_index is not None:
        index_batch = IndexBatch(
            orig_index,
            spans=[len(semantic)],
            counts=[len(orig_index)],
        )

    return adapter(
        cluster_batch,
        semantic_batch,
        prediction_batch,
        index_batch,
    )[0]


def event_products(coords, truth, cluster_coords=None, cluster_features=None):
    """Build event-level semantic and compact cluster label products."""
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
    truth = np.asarray(truth, dtype=np.float32)
    semantic = TensorData(truth, coords=coords)
    if cluster_coords is None:
        cluster_coords = np.empty((0, 3), dtype=np.float32)
    if cluster_features is None:
        cluster_features = np.empty((0, 2), dtype=np.float32)
    clusters = ClusterLabelData(
        coords=np.asarray(cluster_coords, dtype=np.float32),
        features=np.asarray(cluster_features, dtype=np.float32),
    )
    return clusters, semantic


def test_adapter_empty_dummy_deghost_and_validation_paths():
    """Direct event adaptation should handle each empty and alignment contract."""
    adapter = adapter_for_numpy()
    clusters, semantic = event_products([], [])
    adapted = adapt_event(adapter, clusters, semantic, np.empty(0, dtype=int))
    assert adapted.data.shape == (0, 5)

    clusters, semantic = event_products([[0, 0, 0]], [0])
    adapted = adapt_event(
        adapter,
        clusters,
        semantic,
        np.empty(0, dtype=int),
        np.empty(0, dtype=int),
    )
    assert adapted.data.shape == (0, 5)
    dummy = adapt_event(adapter, clusters, semantic, np.array([0]))
    assert np.all(dummy.features == -1)

    clusters, semantic = event_products(
        [[0, 0, 0]], [0], [[0, 0, 0], [1, 0, 0]], [[1, 0], [1, 1]]
    )
    with pytest.raises(ValueError, match="exactly"):
        adapt_event(adapter, clusters, semantic, np.array([0]))

    # Equal row counts must not hide a disagreement in voxel ordering.
    clusters, semantic = event_products(
        [[0, 0, 0], [1, 0, 0]],
        [0, 0],
        [[1, 0, 0], [0, 0, 0]],
        [[1, 0], [1, 1]],
    )
    with pytest.raises(ValueError, match="coordinates must match"):
        adapt_event(adapter, clusters, semantic, np.array([0, 0]))

    # Feature-only products cannot satisfy the voxel-alignment contract.
    semantic = TensorData(np.array([0], dtype=np.float32), feats_only=True)
    with pytest.raises(ValueError, match="requires voxel coordinates"):
        adapt_event(adapter, clusters, semantic, np.array([0]))


def test_adapter_numpy_deghost_expansion_and_no_compatible_cluster():
    """NumPy adaptation should expand predictions and retain deghosted coordinates."""
    adapter = adapter_for_numpy()
    clusters, semantic = event_products(
        [[0, 0, 0], [1, 0, 0]],
        [0, 5],
        [[0, 0, 0]],
        [[1, 0]],
    )
    adapted = adapt_event(
        adapter,
        clusters,
        semantic,
        np.array([0]),
        np.array([0], dtype=np.int64),
    )
    assert adapted.data.shape == (1, 5)

    # A track prediction has no compatible true shower coordinate to borrow.
    clusters, semantic = event_products([[0, 0, 0]], [0], [[0, 0, 0]], [[1, 0]])
    adapted = adapt_event(adapter, clusters, semantic, np.array([1]))
    assert adapted.features[0, 1] == -1


def test_adapter_propagates_unique_touching_instance_target():
    """A unique compatible association should propagate across false positives."""
    adapter = adapter_for_numpy()
    clusters, semantic = event_products(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        [0, 5, 5],
        [[0, 0, 0]],
        [[4, 7]],
    )

    adapted = adapt_event(adapter, clusters, semantic, np.array([0, 0, 0]))

    # The second false positive is reached on the following wavefront step.
    assert len(np.unique(adapted.features[:, 1])) == 1
    np.testing.assert_array_equal(adapted.features[:, 0], [4, 4, 4])


def test_weighted_propagation_distinguishes_face_and_corner_contacts():
    """A face-adjacent owner should beat a corner-adjacent owner by default."""
    coords = [[0, 0, 0], [1, 0, 0], [2, 1, 1]]
    truth = [0, 5, 0]
    prediction = np.zeros(3, dtype=np.int64)
    sources = [(0, 10), (2, 20)]
    clusters, semantic = event_products(
        coords,
        truth,
        np.asarray(coords)[[0, 2]],
        [[1, 10], [2, 20]],
    )

    weighted = adapt_event(
        ClusterLabelAdapter(break_classes=[]),
        clusters,
        semantic,
        prediction,
    )
    unweighted = adapt_event(
        ClusterLabelAdapter(break_classes=[], weighted=False),
        clusters,
        semantic,
        prediction,
    )

    assert sources[0][1] == weighted.cluster_ids[1]
    assert unweighted.cluster_ids[1] == -1


def test_propagation_carries_ambiguity_through_bottleneck():
    """A tied wavefront must remain traversable without acquiring an owner."""
    query_coords = np.asarray([[1, 0, 0], [1, 1, 0], [1, 2, 0]], dtype=np.float32)
    source_coords = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    source_labels = np.asarray([10, 20], dtype=np.int64)

    source_index, distances, ambiguous = PropagationAdapter.propagate(
        query_coords, source_coords, source_labels
    )

    # The final voxel is reachable only through the ambiguous front. It must
    # therefore be reached at distance two rather than stranded as an orphan.
    np.testing.assert_allclose(distances, [1.0, np.sqrt(2), 1.0 + np.sqrt(2)])
    assert np.all(source_index >= 0)
    assert np.all(ambiguous)


def test_propagation_does_not_confuse_same_instance_seeds():
    """Multiple equidistant voxels from one instance are not ambiguous."""
    query_coords = np.asarray([[1, 0, 0], [1, 1, 0]], dtype=np.float32)
    source_coords = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    source_labels = np.asarray([10, 10], dtype=np.int64)

    source_index, distances, ambiguous = PropagationAdapter.propagate(
        query_coords, source_coords, source_labels
    )

    np.testing.assert_allclose(distances, [1.0, np.sqrt(2)])
    assert np.all(source_index >= 0)
    assert not np.any(ambiguous)


def test_propagation_handles_empty_and_unreachable_domains():
    """Propagation should preserve invalid state when no front reaches a voxel."""
    empty = np.empty((0, 3), dtype=np.float32)
    source_coords = np.asarray([[0, 0, 0]], dtype=np.float32)
    source_labels = np.asarray([10], dtype=np.int64)

    source_index, distances, ambiguous = PropagationAdapter.propagate(
        empty, source_coords, source_labels
    )
    assert len(source_index) == len(distances) == len(ambiguous) == 0

    source_index, distances, ambiguous = PropagationAdapter.propagate(
        source_coords, empty, np.empty(0, dtype=np.int64)
    )
    np.testing.assert_array_equal(source_index, [-1])
    np.testing.assert_array_equal(distances, [-1])
    assert not np.any(ambiguous)

    query_coords = np.asarray([[10, 10, 10]], dtype=np.float32)
    source_index, distances, ambiguous = PropagationAdapter.propagate(
        query_coords, source_coords, source_labels
    )
    np.testing.assert_array_equal(source_index, [-1])
    np.testing.assert_array_equal(distances, [-1])
    assert not np.any(ambiguous)


def owner_distances(query_coords, source_coords, adjacent):
    """Compute graph distances from one owner's source voxel set."""
    distances = np.full(len(query_coords), -1, dtype=np.int64)
    touching = np.any(
        np.max(np.abs(query_coords[:, None] - source_coords[None, :]), axis=2) <= 1,
        axis=1,
    )
    queue = list(np.where(touching)[0])
    distances[touching] = 1

    # A list-backed queue is sufficient for these deliberately small oracle
    # problems and keeps the breadth-first traversal independent of production.
    for query_index in queue:
        for neighbor in np.where(adjacent[query_index])[0]:
            if distances[neighbor] < 0:
                distances[neighbor] = distances[query_index] + 1
                queue.append(neighbor)

    return distances


def reference_propagation(query_coords, source_coords, source_labels):
    """Compute exact per-owner graph distances for a small test point cloud."""
    owners = np.unique(source_labels)
    adjacent = (
        np.max(np.abs(query_coords[:, None] - query_coords[None, :]), axis=2) <= 1
    )
    np.fill_diagonal(adjacent, False)

    distances_by_owner = np.stack(
        [
            owner_distances(
                query_coords,
                source_coords[source_labels == owner],
                adjacent,
            )
            for owner in owners
        ]
    )
    masked = np.where(
        distances_by_owner >= 0,
        distances_by_owner,
        np.iinfo(np.int64).max,
    )
    distances = np.min(masked, axis=0)
    reached = distances < np.iinfo(np.int64).max
    ambiguous = np.sum(masked == distances, axis=0) > 1
    closest_owner = np.argmin(masked, axis=0)
    return (
        owners[closest_owner],
        np.where(reached, distances, -1),
        ambiguous & reached,
    )


def test_propagation_matches_per_owner_shortest_path_oracle():
    """Wavefront traversal should reproduce exact geodesic Voronoi ownership."""
    rng = np.random.default_rng(13)
    lattice = np.stack(
        np.meshgrid(np.arange(5), np.arange(4), np.arange(3), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)

    for _ in range(30):
        selection = rng.choice(len(lattice), size=24, replace=False)
        query_coords = lattice[selection].astype(np.float32)
        remaining = np.delete(lattice, selection, axis=0)
        source_selection = rng.choice(len(remaining), size=5, replace=False)
        source_coords = remaining[source_selection].astype(np.float32)
        source_labels = rng.integers(0, 3, size=5, dtype=np.int64)

        source_index, distances, ambiguous = PropagationAdapter.propagate(
            query_coords,
            source_coords,
            source_labels,
            weighted=False,
        )
        owners, expected_distances, expected_ambiguous = reference_propagation(
            query_coords, source_coords, source_labels
        )

        reached = source_index >= 0
        np.testing.assert_array_equal(distances, expected_distances)
        np.testing.assert_array_equal(ambiguous, expected_ambiguous)
        np.testing.assert_array_equal(
            source_labels[source_index[reached & ~ambiguous]],
            owners[reached & ~ambiguous],
        )


def test_adapter_invalidates_exact_instance_ties():
    """Equal-distance associations to distinct instances should stay invalid."""
    adapter = adapter_for_numpy()
    clusters, semantic = event_products(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 2, 0]],
        [0, 5, 0, 5, 5],
        [[0, 0, 0], [2, 0, 0]],
        [[4, 10], [5, 20]],
    )

    adapted = adapt_event(
        adapter,
        clusters,
        semantic,
        np.zeros(5, dtype=np.int64),
    )

    # Both true instance seeds survive, while the tied connecting front does
    # not provide an arbitrary target at any propagated depth.
    assert np.all(adapted.features[[0, 2], 1] >= 0)
    assert np.all(adapted.features[[1, 3, 4]] == -1)


def test_adapter_splits_disconnected_effective_target():
    """Disconnected pieces of one truth instance should receive distinct IDs."""
    adapter = adapter_for_numpy()
    clusters, semantic = event_products(
        [[0, 0, 0], [2, 0, 0]],
        [0, 0],
        [[0, 0, 0], [2, 0, 0]],
        [[3, 9], [5, 9]],
    )

    adapted = adapt_event(adapter, clusters, semantic, np.array([0, 0]))

    assert np.all(adapted.features[:, 1] >= 0)
    assert len(np.unique(adapted.features[:, 1])) == 2
    np.testing.assert_array_equal(adapted.features[:, 0], [3, 5])
