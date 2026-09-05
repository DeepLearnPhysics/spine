"""Tests for GrapPA graph augmentations."""

import numpy as np
import pytest
import torch

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch, TensorSchema
from spine.model.grappa.augment import (
    EdgeDropout,
    EdgeSelection,
    FeatureMask,
    FeatureNoise,
    NodeDropout,
    NodeSelection,
)


def test_edge_dropout_validates_probability() -> None:
    """Reject probabilities outside the closed unit interval."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        EdgeDropout(-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        EdgeDropout(1.1)


def test_edge_dropout_keeps_reciprocal_pairs(monkeypatch) -> None:
    """Sample one decision per adjacent undirected edge pair and event."""
    edge_index = EdgeIndexBatch(
        np.array(
            [
                [0, 1, 0, 2, 3, 4],
                [1, 0, 2, 0, 4, 3],
            ]
        ),
        counts=[4, 2],
        spans=[3, 2],
        directed=False,
    )
    samples = iter((np.array([0.2, 0.8]), np.array([0.9])))
    monkeypatch.setattr(np.random, "random", lambda _: next(samples))

    selection = EdgeDropout(0.5)(edge_index)
    filtered = selection.filter_edge_index(edge_index)
    keep = selection.keep

    np.testing.assert_array_equal(keep.data, [False, False, True, True, True, True])
    np.testing.assert_array_equal(filtered.counts, [2, 2])
    np.testing.assert_array_equal(
        filtered.index,
        np.array([[0, 2, 3, 4], [2, 0, 4, 3]]),
    )


def test_edge_dropout_samples_directed_edges_on_original_backend(monkeypatch) -> None:
    """Directed edges are independent and Torch graphs stay Torch-backed."""
    edge_index = EdgeIndexBatch(
        torch.tensor([[0, 1, 2], [1, 2, 0]]),
        counts=[3, 0],
        spans=[3, 1],
        directed=True,
    )
    samples = iter((np.array([0.1, 0.6, 0.9]), np.empty(0)))
    monkeypatch.setattr(np.random, "random", lambda _: next(samples))

    selection = EdgeDropout(0.5)(edge_index)
    filtered = selection.filter_edge_index(edge_index)
    keep = selection.keep

    assert isinstance(filtered.index, torch.Tensor)
    assert filtered.counts.tolist() == [2, 0]
    assert filtered.index.tolist() == [[1, 2], [2, 0]]
    np.testing.assert_array_equal(keep.data, [False, True, True])


@pytest.mark.parametrize(
    "index, match",
    [
        (np.array([[0], [1]]), "even edge count"),
        (
            np.array([[0, 1], [1, 2]]),
            "adjacent reciprocal edge pairs",
        ),
    ],
)
def test_edge_dropout_rejects_malformed_undirected_graphs(index, match) -> None:
    """Materialized undirected inputs must honor GrapPA's pair convention."""
    edge_index = EdgeIndexBatch(
        index, counts=[index.shape[1]], spans=[3], directed=True
    )
    edge_index.directed = False

    with pytest.raises(ValueError, match=match):
        EdgeDropout(0.5)(edge_index)


def test_filter_tensor_batch_preserves_metadata_and_empty_events() -> None:
    """Selections recompute counts while retaining the logical tensor schema."""
    batch = TensorBatch(torch.arange(8).reshape(4, 2), counts=[2, 0, 2])
    keep = TensorBatch(
        torch.tensor([True, False, False, True]),
        counts=[2, 0, 2],
    )

    filtered = EdgeSelection(keep).filter_tensor(batch)

    assert filtered.counts.tolist() == [1, 0, 1]
    assert filtered.data.tolist() == [[0, 1], [6, 7]]
    assert filtered.schema == batch.schema

    with pytest.raises(ValueError, match="must align"):
        EdgeSelection(TensorBatch(np.ones(4), counts=[1, 3])).filter_tensor(batch)

    empty = TensorBatch(torch.empty((0, 2)), counts=[0, 0])
    empty_keep = TensorBatch(np.empty(0, dtype=bool), counts=[0, 0])
    assert EdgeSelection(empty_keep).filter_tensor(empty).counts.tolist() == [0, 0]

    with pytest.raises(ValueError, match="one-dimensional"):
        EdgeSelection(TensorBatch(np.ones((2, 1)), counts=[2]))


def test_edge_selection_composes_original_axis_masks() -> None:
    """Sequential graph changes collapse to one original-edge selection."""
    first = EdgeSelection(
        TensorBatch(np.array([True, False, True, True]), counts=[3, 1])
    )
    second = EdgeSelection(TensorBatch(np.array([False, True, True]), counts=[2, 1]))

    combined = first.compose(second)

    np.testing.assert_array_equal(combined.keep.data, [False, False, True, True])
    np.testing.assert_array_equal(combined.counts, [1, 1])

    with pytest.raises(ValueError, match="must align"):
        first.compose(EdgeSelection(TensorBatch(np.ones(3), counts=[1, 2])))


def test_feature_mask_samples_named_columns_per_event(monkeypatch) -> None:
    """Each event shares a decision for every resolved feature column."""
    schema = TensorSchema(
        feature_fields={"position": (0, 1), "value": (2,)}, feats_only=True
    )
    batch = TensorBatch(np.arange(15.0).reshape(5, 3), counts=[2, 3], schema=schema)
    monkeypatch.setattr(
        np.random,
        "random",
        lambda shape: np.array([[0.1, 0.9], [0.8, 0.2]]),
    )

    result = FeatureMask(0.5, columns="position", fill_value=-1.0)(batch)

    np.testing.assert_array_equal(
        result.data,
        [
            [-1.0, 1.0, 2.0],
            [-1.0, 4.0, 5.0],
            [6.0, -1.0, 8.0],
            [9.0, -1.0, 11.0],
            [12.0, -1.0, 14.0],
        ],
    )
    np.testing.assert_array_equal(batch.data, np.arange(15.0).reshape(5, 3))
    assert result.schema is schema


def test_feature_mask_supports_elementwise_torch_scalars(monkeypatch) -> None:
    """Element masks preserve Torch autograd and scalar feature layout."""
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    batch = TensorBatch(values, counts=[2, 1])
    monkeypatch.setattr(
        np.random, "random", lambda shape: np.array([[0.1], [0.9], [0.2]])
    )

    result = FeatureMask(0.5, granularity="element")(batch)
    result.data.sum().backward()

    assert result.data.tolist() == [0.0, 2.0, 0.0]
    assert values.grad.tolist() == [0.0, 1.0, 0.0]


def test_feature_noise_supports_event_relative_noise(monkeypatch) -> None:
    """Relative event noise applies one column perturbation to all event rows."""
    batch = TensorBatch(np.ones((3, 3)), counts=[2, 1])
    monkeypatch.setattr(
        np.random,
        "normal",
        lambda size: np.array([[1.0, -1.0], [2.0, 0.5]]),
    )

    result = FeatureNoise([0.1, 0.2], columns=[0, -1], mode="relative")(batch)

    np.testing.assert_allclose(
        result.data,
        [[1.1, 1.0, 0.8], [1.1, 1.0, 0.8], [1.2, 1.0, 1.1]],
    )
    np.testing.assert_array_equal(batch.data, np.ones((3, 3)))


def test_feature_noise_supports_elementwise_torch_noise(monkeypatch) -> None:
    """Additive element noise stays on the Torch backend and remains differentiable."""
    values = torch.ones((2, 2), requires_grad=True)
    batch = TensorBatch(values, counts=[2])
    monkeypatch.setattr(
        np.random,
        "normal",
        lambda size: np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    result = FeatureNoise(0.5, granularity="element")(batch)
    result.data.sum().backward()

    assert isinstance(result.data, torch.Tensor)
    assert result.data.tolist() == [[1.5, 2.0], [2.5, 3.0]]
    assert values.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_feature_augmentations_validate_configuration_and_inputs() -> None:
    """Reject invalid selectors, distributions and incompatible features."""
    with pytest.raises(ValueError, match="granularity"):
        FeatureMask(0.1, granularity="batch")
    with pytest.raises(ValueError, match="must not be empty"):
        FeatureMask(0.1, columns=[])
    with pytest.raises(TypeError, match="integer indexes or field names"):
        FeatureMask(0.1, columns=[0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
        FeatureMask(1.1)
    with pytest.raises(ValueError, match="mode"):
        FeatureNoise(0.1, mode="scale")
    with pytest.raises(ValueError, match="finite nonnegative"):
        FeatureNoise([-0.1])
    with pytest.raises(ValueError, match="finite nonnegative"):
        FeatureNoise(np.nan)
    with pytest.raises(ValueError, match="must not be empty"):
        FeatureNoise([])

    batch = TensorBatch(np.ones((2, 2)), counts=[2])
    with pytest.raises(IndexError, match="out of bounds"):
        FeatureMask(0.1, columns=2)(batch)
    with pytest.raises(KeyError, match="Unknown feature field"):
        FeatureMask(0.1, columns="missing")(batch)
    with pytest.raises(ValueError, match="one value per"):
        FeatureNoise([0.1, 0.2], columns=[0])(batch)
    with pytest.raises(TypeError, match="floating-point"):
        FeatureNoise(0.1)(TensorBatch(np.ones((2, 2), dtype=np.int64), counts=[2]))

    torch_batch = TensorBatch(torch.ones((1, 1)), counts=[1])
    assert FeatureMask(0.0)(torch_batch).data.tolist() == [[1.0]]


def test_node_dropout_samples_individual_nodes_per_event(monkeypatch) -> None:
    """Independent dropout respects event boundaries and keeps one node."""
    samples = iter((np.array([0.1, 0.8, 0.2]), np.array([0.1, 0.2])))
    monkeypatch.setattr(np.random, "random", lambda _: next(samples))
    monkeypatch.setattr(np.random, "randint", lambda _: 1)

    selection = NodeDropout(0.5)([3, 2])

    np.testing.assert_array_equal(
        selection.keep.data, [False, True, False, False, True]
    )
    np.testing.assert_array_equal(selection.counts, [1, 1])


def test_node_dropout_samples_complete_groups_and_retains_invalid(monkeypatch) -> None:
    """Grouped dropout shares decisions and never merges invalid IDs."""
    groups = TensorBatch(np.array([2, 2, 4, -1, -1, 7]), counts=[5, 1])
    samples = iter((np.array([0.2, 0.8]), np.array([0.1])))
    monkeypatch.setattr(np.random, "random", lambda _: next(samples))
    monkeypatch.setattr(np.random, "randint", lambda _: 0)

    selection = NodeDropout(0.5, group_by="ancestor")([5, 1], groups)

    np.testing.assert_array_equal(
        selection.keep.data, [False, False, True, True, True, True]
    )
    np.testing.assert_array_equal(selection.counts, [3, 1])


def test_node_dropout_limits_sampling_to_eligible_nodes(monkeypatch) -> None:
    """Static eligibility protects unselected nodes from independent dropout."""
    eligible = TensorBatch(np.array([False, True, True, False]), counts=[4])
    monkeypatch.setattr(np.random, "random", lambda _: np.array([0.2, 0.8]))

    selection = NodeDropout(0.5, select={"shape": ["michel", "delta"]})(
        [4], eligible=eligible
    )

    np.testing.assert_array_equal(selection.keep.data, [True, False, True, True])


@pytest.mark.parametrize(
    "group_match, expected",
    [("any", [False, False, False, False]), ("all", [True, True, True, True])],
)
def test_grouped_node_dropout_respects_selection_match_policy(
    monkeypatch, group_match, expected
) -> None:
    """Group eligibility can require any or every member to match selection."""
    groups = TensorBatch(np.array([0, 0, 1, 1]), counts=[4])
    eligible = TensorBatch(np.array([True, False, True, False]), counts=[4])
    monkeypatch.setattr(np.random, "random", lambda size: np.zeros(size))

    selection = NodeDropout(
        1.0,
        group_by="group",
        select={"group_primary": 0},
        group_match=group_match,
        keep_one=False,
    )([4], groups, eligible)

    np.testing.assert_array_equal(selection.keep.data, expected)


def test_node_dropout_builds_compound_live_eligibility(
    graph_labels, graph_clusters
) -> None:
    """Live selectors combine categorical fields with AND semantics."""
    dropout = NodeDropout(
        0.1,
        select={"shape": "track", "pid": "electron"},
    )

    eligible = dropout.build_eligibility(graph_labels, graph_clusters)

    np.testing.assert_array_equal(eligible.data, [False, True, False])
    with pytest.raises(ValueError, match="without a `select` mapping"):
        NodeDropout(0.1).build_eligibility(graph_labels, graph_clusters)


def test_node_dropout_validates_configuration_and_groups() -> None:
    """Reject invalid probabilities, counts and grouped-input contracts."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        NodeDropout(-0.1)
    with pytest.raises(ValueError, match="nonempty"):
        NodeDropout(0.1, group_by="")
    with pytest.raises(ValueError, match="must be 'any' or 'all'"):
        NodeDropout(0.1, group_match="some")
    with pytest.raises(ValueError, match="requires grouped"):
        NodeDropout(0.1, group_match="all")
    with pytest.raises(ValueError, match="must not be empty"):
        NodeDropout(0.1, select={})
    with pytest.raises(ValueError, match="must not be empty"):
        NodeDropout(0.1, select={"shape": []})
    with pytest.raises(ValueError, match="String values are not supported"):
        NodeDropout(0.1, select={"group_primary": "secondary"})
    with pytest.raises(ValueError, match="nonnegative one-dimensional"):
        NodeDropout(0.1)([-1])
    with pytest.raises(ValueError, match="requires node-aligned"):
        NodeDropout(0.1, group_by="group")([2])
    with pytest.raises(ValueError, match="must align"):
        NodeDropout(0.1, group_by="group")(
            [2], TensorBatch(np.array([[0], [1]]), counts=[2])
        )
    with pytest.raises(ValueError, match="node_dropout_eligible"):
        NodeDropout(0.1, select={"primary": 0})([2])
    with pytest.raises(ValueError, match="eligibility must align"):
        NodeDropout(0.1, select={"primary": 0})(
            [2], eligible=TensorBatch(np.array([[True], [False]]), counts=[2])
        )
    dropout = NodeDropout(0.1, select={"primary": 0, "type": "photon"})
    assert dropout.select is not None
    assert set(dropout.select) == {"group_primary", "pid"}
    np.testing.assert_array_equal(dropout.select["pid"], [0])

    selection = NodeDropout(0.0)(torch.tensor([1]))
    np.testing.assert_array_equal(selection.keep.data, [True])


def test_node_selection_filters_clusters_and_remaps_edges() -> None:
    """Dropping nodes removes incident edges and compacts every namespace."""
    selection = NodeSelection(
        TensorBatch(np.array([True, False, True, False, True]), counts=[3, 2])
    )
    edge_index = EdgeIndexBatch(
        torch.tensor([[0, 1, 0, 3], [1, 2, 2, 4]]),
        counts=[3, 1],
        spans=[3, 2],
        directed=True,
    )
    clusters = IndexBatch(
        [
            np.array([0]),
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([5]),
            np.array([6, 7]),
        ],
        spans=[5, 3],
        counts=[3, 2],
        single_counts=[1, 2, 2, 1, 2],
    )

    filtered_edges, edge_selection = selection.filter_edge_index(edge_index)
    filtered_clusters = selection.filter_index(clusters)

    assert isinstance(filtered_edges.index, torch.Tensor)
    assert filtered_edges.index.tolist() == [[0], [1]]
    assert filtered_edges.counts.tolist() == [1, 0]
    assert filtered_edges.spans.tolist() == [2, 1]
    np.testing.assert_array_equal(edge_selection.keep.data, [False, False, True, False])
    np.testing.assert_array_equal(filtered_clusters.counts, [2, 1])
    np.testing.assert_array_equal(filtered_clusters.single_counts, [1, 2, 2])

    flat = IndexBatch(np.arange(5), spans=[3, 2], counts=[3, 2])
    np.testing.assert_array_equal(selection.filter_index(flat).index, [0, 2, 4])

    with pytest.raises(ValueError, match="must align"):
        selection.filter_edge_index(
            EdgeIndexBatch(np.empty((2, 0), dtype=np.int64), [0], [5], True)
        )
