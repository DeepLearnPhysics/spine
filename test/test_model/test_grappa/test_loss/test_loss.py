"""Focused tests for GNN loss-label construction."""

import numpy as np
import pytest
import torch

from spine.data import (
    ClusterLabelBatch,
    EdgeIndexBatch,
    IndexBatch,
    Meta,
    TensorBatch,
    TensorSchema,
)
from spine.model.grappa.evaluation import edge_assignment_forest_batch
from spine.model.grappa.loss import (
    EdgeChannelLoss,
    NodeClassLoss,
    NodeOrientLoss,
    NodeRegressionLoss,
    NodeShowerPrimaryLoss,
    NodeVertexLoss,
)


def test_forest_assignment_builds_valid_spanning_tree_labels():
    edge_index = EdgeIndexBatch(
        np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int64),
        counts=np.array([3], dtype=np.int64),
        spans=np.array([3], dtype=np.int64),
        directed=True,
    )
    edge_prediction = TensorBatch(
        np.array(
            [
                [0.0, 3.0],
                [3.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        ),
        counts=np.array([3], dtype=np.int64),
    )
    group_ids = TensorBatch(
        np.zeros(3, dtype=np.int64),
        counts=np.array([3], dtype=np.int64),
    )

    assignments, valid_mask = edge_assignment_forest_batch(
        edge_index,
        edge_prediction,
        group_ids,
    )

    assert assignments.numpy_tensor().sum() == 2
    assert valid_mask.numpy_tensor().sum() == 2


def test_vertex_loss_normalizes_each_batch_entry_with_its_own_meta():
    """Normalize vertex labels with the image dimensions of their entry."""
    data = TensorBatch(
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        ),
        counts=[1, 1],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    particles = {
        "interaction_primary": TensorBatch(np.ones(2, dtype=np.int64), counts=[1, 1]),
        "vertex": TensorBatch(
            np.array(
                [
                    [50.0, 25.0, 10.0],
                    [100.0, 50.0, 20.0],
                ],
                dtype=np.float32,
            ),
            counts=[1, 1],
        ),
    }
    clust_label = ClusterLabelBatch(data, particles)
    clusts = IndexBatch(
        [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ],
        spans=[1, 1],
        counts=[1, 1],
        single_counts=[1, 1],
    )
    node_pred = TensorBatch(
        torch.tensor(
            [
                [0.0, 1.0, 0.5, 0.5, 0.5],
                [0.0, 1.0, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        ),
        counts=[1, 1],
    )
    meta = [
        Meta(
            lower=np.zeros(3),
            upper=np.array([100.0, 50.0, 20.0]),
            size=np.ones(3),
            count=np.array([100, 50, 20]),
        ),
        Meta(
            lower=np.zeros(3),
            upper=np.array([200.0, 100.0, 40.0]),
            size=np.ones(3),
            count=np.array([200, 100, 40]),
        ),
    ]

    result = NodeVertexLoss(
        only_contained=False,
        normalize_positions=True,
        return_vertex_labels=True,
    )(clust_label, clusts, node_pred, meta=meta)

    torch.testing.assert_close(result["reg_loss"], torch.tensor(0.0))
    np.testing.assert_allclose(
        result["labels"].numpy_tensor(),
        np.full((2, 3), 0.5),
    )


def test_node_regression_loss_masks_invalid_targets(graph_labels, graph_clusters):
    """Regression ignores unavailable labels and reports relative spread."""
    prediction = TensorBatch(
        torch.tensor([[1.0], [20.0], [6.0]]),
        graph_clusters.counts,
    )

    loss_fn = NodeRegressionLoss(target="energy")
    result = loss_fn(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
        return_target=True,
    )

    assert result["count"] == 2
    torch.testing.assert_close(result["loss"], torch.tensor(2.5))
    assert result["accuracy"] == pytest.approx(0.5)
    cached = loss_fn(
        prediction,
        clusts=graph_clusters,
        labels=result["target"],
        valid_mask=result["valid"],
    )
    torch.testing.assert_close(cached["loss"], result["loss"])

    incompatible = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)
    with pytest.raises(ValueError, match="incompatible numbers"):
        NodeRegressionLoss(target="energy")(
            incompatible,
            clust_label=graph_labels,
            clusts=graph_clusters,
        )


def test_node_regression_loss_supports_vector_targets(graph_labels, graph_clusters):
    """Vector regression applies one validity decision to each node."""
    prediction = TensorBatch(torch.zeros((3, 3)), graph_clusters.counts)

    result = NodeRegressionLoss(target="vertex")(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )

    assert result["count"] == 3
    torch.testing.assert_close(result["loss"], torch.tensor(95.0))


def test_node_regression_loss_handles_no_valid_targets(graph_labels, graph_clusters):
    """An entirely unavailable regression target yields a neutral result."""
    graph_labels.particles["energy"].data.fill(-1.0)
    prediction = TensorBatch(torch.zeros((3, 1)), graph_clusters.counts)

    result = NodeRegressionLoss(target="energy")(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )

    assert result["count"] == 0
    assert result["accuracy"] == 1.0
    torch.testing.assert_close(result["loss"], torch.tensor(0.0))


def test_cached_node_target_validation(graph_clusters):
    """Cached target pairs must be typed, aligned and one-dimensionally masked."""
    prediction = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)
    labels = TensorBatch(np.zeros(3, dtype=np.int64), graph_clusters.counts)
    valid = TensorBatch(np.ones(3, dtype=bool), graph_clusters.counts)
    loss_fn = NodeClassLoss(target="pid")

    with pytest.raises(ValueError, match="provided together"):
        loss_fn(prediction, clusts=graph_clusters, labels=labels)
    with pytest.raises(TypeError, match="must be TensorBatch"):
        loss_fn(
            prediction,
            clusts=graph_clusters,
            labels=np.zeros(3),
            valid_mask=valid,
        )
    with pytest.raises(ValueError, match="labels must align"):
        loss_fn(
            prediction,
            clusts=graph_clusters,
            labels=TensorBatch(np.zeros(3), counts=[3]),
            valid_mask=valid,
        )
    with pytest.raises(ValueError, match="validity mask must align"):
        loss_fn(
            prediction,
            clusts=graph_clusters,
            labels=labels,
            valid_mask=TensorBatch(np.ones(3), counts=[3]),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        loss_fn(
            prediction,
            clusts=graph_clusters,
            labels=labels,
            valid_mask=TensorBatch(np.ones((3, 1)), graph_clusters.counts),
        )
    with pytest.raises(ValueError, match="structured cluster labels"):
        loss_fn(prediction, clusts=graph_clusters)

    result = loss_fn(
        prediction,
        clusts=graph_clusters,
        labels=labels,
        valid_mask=valid,
        return_target=True,
    )
    assert result["count"] == 3
    assert result["target"] is labels
    assert result["valid"] is valid


def test_other_cached_target_pairs_validate_required_inputs(graph_clusters):
    """Every cacheable node and edge objective should enforce the same pair."""
    node_pred = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)
    node_labels = TensorBatch(np.zeros(3), graph_clusters.counts)
    edge_index = EdgeIndexBatch(
        np.array([[0], [1]], dtype=np.int64),
        counts=[1, 0],
        spans=graph_clusters.counts,
        directed=True,
    )
    edge_pred = TensorBatch(torch.zeros((1, 2)), counts=[1, 0])
    edge_labels = TensorBatch(np.zeros(1), counts=[1, 0])

    with pytest.raises(ValueError, match="provided together"):
        NodeRegressionLoss(target="energy")(
            node_pred, clusts=graph_clusters, labels=node_labels
        )
    with pytest.raises(ValueError, match="structured cluster labels"):
        NodeRegressionLoss(target="energy")(node_pred, clusts=graph_clusters)

    with pytest.raises(ValueError, match="provided together"):
        NodeShowerPrimaryLoss()(node_pred, clusts=graph_clusters, labels=node_labels)
    with pytest.raises(ValueError, match="structured cluster labels"):
        NodeShowerPrimaryLoss()(node_pred, clusts=graph_clusters)

    with pytest.raises(ValueError, match="provided together"):
        NodeOrientLoss()(
            node_pred,
            clusts=graph_clusters,
            labels=node_labels,
        )
    with pytest.raises(ValueError, match="endpoint inputs"):
        NodeOrientLoss()(node_pred, clusts=graph_clusters)

    edge_loss = EdgeChannelLoss(target="group")
    with pytest.raises(ValueError, match="provided together"):
        edge_loss(
            edge_index,
            edge_pred,
            clusts=graph_clusters,
            labels=edge_labels,
        )
    with pytest.raises(ValueError, match="structured cluster labels"):
        edge_loss(edge_index, edge_pred, clusts=graph_clusters)

    edge_valid = TensorBatch(np.ones(1, dtype=bool), counts=[1, 0])
    result = edge_loss(
        edge_index,
        edge_pred,
        clusts=graph_clusters,
        labels=edge_labels,
        valid_mask=edge_valid,
        return_target=True,
    )
    assert result["count"] == 1
    assert result["target"] is edge_labels
    assert result["valid"] is edge_valid


def test_cached_grappa_losses_do_not_require_clusters():
    """Cached node and edge objectives run from aligned graph products alone."""
    node_pred = TensorBatch(torch.zeros((3, 2)), counts=[2, 1])
    node_labels = TensorBatch(torch.zeros(3), counts=[2, 1])
    node_valid = TensorBatch(torch.ones(3), counts=[2, 1])
    edge_index = EdgeIndexBatch(
        torch.tensor([[0], [1]], dtype=torch.long),
        counts=[1, 0],
        spans=[2, 1],
        directed=True,
    )
    edge_pred = TensorBatch(torch.zeros((1, 2)), counts=[1, 0])
    edge_labels = TensorBatch(torch.zeros(1), counts=[1, 0])
    edge_valid = TensorBatch(torch.ones(1), counts=[1, 0])

    class_result = NodeClassLoss(target="pid")(
        node_pred=node_pred,
        labels=node_labels,
        valid_mask=node_valid,
    )
    primary_result = NodeShowerPrimaryLoss()(
        node_pred=node_pred,
        labels=node_labels,
        valid_mask=node_valid,
    )
    orient_result = NodeOrientLoss()(
        node_pred=node_pred,
        labels=node_labels,
        valid_mask=node_valid,
    )
    edge_result = EdgeChannelLoss(target="group")(
        edge_index=edge_index,
        edge_pred=edge_pred,
        labels=edge_labels,
        valid_mask=edge_valid,
    )

    assert class_result["count"] == 3
    assert primary_result["count"] == 3
    assert orient_result["count"] == 3
    assert edge_result["count"] == 1

    with pytest.raises(TypeError, match="node_pred"):
        NodeClassLoss(target="pid")()
    with pytest.raises(TypeError, match="node_pred"):
        NodeRegressionLoss(target="energy")()
    with pytest.raises(TypeError, match="node_pred"):
        NodeShowerPrimaryLoss()()
    with pytest.raises(TypeError, match="node_pred"):
        NodeOrientLoss()()
    with pytest.raises(TypeError, match="edge_index.*edge_pred"):
        EdgeChannelLoss(target="group")()


def test_cached_forest_loss_uses_edge_index_node_spans():
    """Validate forest node targets without an IndexBatch cluster product."""
    edge_index = EdgeIndexBatch(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        counts=[2],
        spans=[2],
        directed=True,
    )
    edge_pred = TensorBatch(torch.zeros((2, 2)), counts=[2])
    group_ids = TensorBatch(torch.zeros(2), counts=[2])
    valid_mask = TensorBatch(torch.ones(2), counts=[2])

    result = EdgeChannelLoss(target="group", mode="forest")(
        edge_index=edge_index,
        edge_pred=edge_pred,
        labels=group_ids,
        valid_mask=valid_mask,
    )
    assert result["count"] > 0

    bad_group_ids = TensorBatch(torch.zeros(2), counts=[1, 1])
    with pytest.raises(ValueError, match="align with graph nodes"):
        EdgeChannelLoss(target="group", mode="forest")(
            edge_index=edge_index,
            edge_pred=edge_pred,
            labels=bad_group_ids,
            valid_mask=valid_mask,
        )


def test_node_losses_filter_low_quality_clusters(graph_labels):
    """Node classification and regression share overlap-quality filtering."""
    objects = _mixed_graph_clusters()
    class_prediction = TensorBatch(torch.zeros((3, 2)), objects.counts)
    class_result = NodeClassLoss(
        target="pid",
        min_purity=[0.75, 0.75],
    )(class_prediction, clust_label=graph_labels, clusts=objects)
    assert class_result["count"] == 1
    assert class_result["count_rejected"] == 2

    graph_labels.particles["energy"].data[1] = 6.0
    reg_prediction = TensorBatch(torch.zeros((3, 1)), objects.counts)
    reg_result = NodeRegressionLoss(
        target="energy",
        min_purity=[0.75, 0.75],
        quality_num_classes=2,
    )(reg_prediction, clust_label=graph_labels, clusts=objects)
    assert reg_result["count"] == 1
    assert reg_result["count_rejected"] == 2


def test_node_overlap_thresholds_validate_class_counts(graph_labels, graph_clusters):
    """Node threshold vectors must match the classification domain."""
    with pytest.raises(ValueError, match="requires `num_classes`"):
        NodeRegressionLoss(target="energy", min_iou=[0.5])

    prediction = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)
    with pytest.raises(ValueError, match="exactly 2 values"):
        NodeClassLoss(target="pid", min_iou=[0.5])(
            prediction,
            clust_label=graph_labels,
            clusts=graph_clusters,
        )
    with pytest.raises(ValueError, match="align with clusters"):
        NodeClassLoss(target="pid")(
            prediction,
            clust_label=graph_labels,
            clusts=graph_clusters,
            node_quality_mask=np.array([True]),
        )


def test_vertex_loss_validates_inputs(graph_labels, graph_clusters):
    """Vertex regression rejects malformed outputs and missing context."""
    bad_prediction = TensorBatch(torch.zeros((3, 4)), graph_clusters.counts)
    with pytest.raises(ValueError, match="contain 5 features"):
        NodeVertexLoss(only_contained=False)(
            graph_labels,
            graph_clusters,
            bad_prediction,
        )

    prediction = TensorBatch(torch.zeros((3, 5)), graph_clusters.counts)
    with pytest.raises(ValueError, match="provide `meta`"):
        NodeVertexLoss(only_contained=False, normalize_positions=True)(
            graph_labels,
            graph_clusters,
            prediction,
        )
    with pytest.raises(ValueError, match="particle end points"):
        NodeVertexLoss(only_contained=False, use_anchor_points=True)(
            graph_labels,
            graph_clusters,
            prediction,
        )


def test_vertex_loss_anchors_offsets_to_particle_endpoints(
    graph_labels,
    graph_clusters,
):
    """Anchor mode interprets the regressed values as endpoint offsets."""
    logits = torch.tensor(
        [
            [0.0, 1.0, 0.5, 0.0, 0.0],
            [1.0, 0.0, 9.0, 9.0, 9.0],
            [0.0, 1.0, 0.0, 0.5, 0.0],
        ]
    )
    prediction = TensorBatch(logits, graph_clusters.counts)
    starts = TensorBatch(
        torch.tensor([[0.5, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 7.5, 9.0]]),
        graph_clusters.counts,
    )
    ends = TensorBatch(
        torch.tensor([[20.0, 20.0, 20.0], [8.0, 8.0, 8.0], [30.0, 30.0, 30.0]]),
        graph_clusters.counts,
    )

    result = NodeVertexLoss(only_contained=False, use_anchor_points=True)(
        graph_labels,
        graph_clusters,
        prediction,
        start_points=starts,
        end_points=ends,
    )

    assert result["reg_count"] == 2
    torch.testing.assert_close(result["reg_loss"], torch.tensor(0.0))
    assert result["reg_accuracy"] == 0.0


def test_vertex_loss_validates_metadata_count(graph_labels, graph_clusters):
    """Position normalization requires metadata for every batch entry."""
    prediction = TensorBatch(torch.zeros((3, 5)), graph_clusters.counts)
    meta = [
        Meta(
            lower=np.zeros(3),
            upper=np.ones(3),
            size=np.ones(3),
            count=np.ones(3, dtype=np.int64),
        )
    ]

    with pytest.raises(ValueError, match="one metadata entry"):
        NodeVertexLoss(only_contained=False, normalize_positions=True)(
            graph_labels,
            graph_clusters,
            prediction,
            meta=meta,
        )


def _edge_inputs():
    """Build a minimal two-entry graph and aligned edge logits."""
    edge_index = EdgeIndexBatch(
        np.array([[0, 1], [1, 0]], dtype=np.int64),
        counts=[2, 0],
        spans=[2, 1],
        directed=True,
    )
    edge_pred = TensorBatch(
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
        counts=[2, 0],
    )

    return edge_index, edge_pred


def _mixed_graph_clusters():
    """Return two impure clusters and one pure cluster."""
    return IndexBatch(
        [
            np.array([0, 2], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
            np.array([4, 5], dtype=np.int64),
        ],
        spans=[4, 2],
        counts=[2, 1],
        single_counts=[2, 2, 2],
    )


def _add_particle_fields(graph_labels, **fields):
    """Attach particle fields shared by the focused GNN loss tests."""
    for name, values in fields.items():
        graph_labels.particles[name] = TensorBatch(
            np.asarray(values),
            counts=[2, 1],
        )


def test_edge_channel_group_and_empty_losses(graph_labels, graph_clusters):
    """Edge classification handles weighted, invalid and empty selections."""
    _add_particle_fields(graph_labels, group=[0, 0, 1])
    edge_index, edge_pred = _edge_inputs()

    result = EdgeChannelLoss(target="group", balance_loss=True)(
        edge_index,
        edge_pred,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )
    assert result["count"] == 2
    assert result["accuracy"] == pytest.approx(0.5)

    graph_labels.particles["group"].data.fill(-1)
    result = EdgeChannelLoss(target="group")(
        edge_index,
        edge_pred,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )
    assert result["count"] == 0
    assert result["accuracy"] == 1.0
    torch.testing.assert_close(result["loss"], torch.tensor(0.0))


def test_edge_channel_truth_modes(graph_labels, graph_clusters):
    """Forest modes construct graph-derived targets and validate inputs."""
    _add_particle_fields(graph_labels, group=[0, 0, 1], particle=[0, 1, 0])
    edge_index, edge_pred = _edge_inputs()

    result = EdgeChannelLoss(target="group", mode="forest")(
        edge_index,
        edge_pred,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )
    assert result["count"] == 2

    particle_loss = EdgeChannelLoss(target="group", mode="particle_forest")
    with pytest.raises(ValueError, match="true_edge_index"):
        particle_loss(
            edge_index,
            edge_pred,
            clust_label=graph_labels,
            clusts=graph_clusters,
        )

    true_edge_index = EdgeIndexBatch(
        np.array([[0], [1]], dtype=np.int64),
        counts=[1, 0],
        spans=[2, 1],
        directed=True,
    )
    result = particle_loss(
        edge_index,
        edge_pred,
        clust_label=graph_labels,
        clusts=graph_clusters,
        true_edge_index=true_edge_index,
    )
    assert result["count"] == 2

    with pytest.raises(ValueError, match="not recognized"):
        EdgeChannelLoss(target="group", mode="mystery")(
            edge_index,
            edge_pred,
            clust_label=graph_labels,
            clusts=graph_clusters,
        )


def test_edge_forest_rebuilds_cached_target_from_current_logits(
    graph_labels,
    graph_clusters,
    monkeypatch,
):
    """Cached forest primitives should not freeze an old predicted tree."""
    _add_particle_fields(graph_labels, group=[0, 0, 1])
    edge_index, initial_pred = _edge_inputs()
    current_pred = TensorBatch(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        counts=edge_index.counts,
    )

    def build_forest(_, edge_pred, group_ids):
        """Select visibly different targets from the current logits."""
        initial = edge_pred.numpy_tensor()[0, 0] < 1.0
        labels = np.asarray([1, 0] if initial else [0, 1])
        valid = np.asarray([True, False] if initial else [False, True])
        assert group_ids.shape[0] == len(graph_clusters.index_list)
        return (
            TensorBatch(labels, edge_index.counts),
            TensorBatch(valid, edge_index.counts),
        )

    monkeypatch.setattr(
        "spine.model.grappa.loss.edge_channel.edge_assignment_forest_batch",
        build_forest,
    )
    loss_fn = EdgeChannelLoss(target="group", mode="forest")
    cached_inputs = loss_fn(
        edge_index,
        initial_pred,
        clust_label=graph_labels,
        clusts=graph_clusters,
        return_target=True,
    )
    assert cached_inputs["target"].shape[0] == len(graph_clusters.index_list)
    np.testing.assert_array_equal(
        cached_inputs["valid"].numpy_tensor(),
        np.ones(2, dtype=bool),
    )

    live = loss_fn(
        edge_index,
        current_pred,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )
    cached = loss_fn(
        edge_index,
        current_pred,
        clusts=graph_clusters,
        labels=cached_inputs["target"],
        valid_mask=cached_inputs["valid"],
    )
    assert live["count"] == cached["count"] == 1
    torch.testing.assert_close(cached["loss"], live["loss"])

    # Cached batches may carry tensor-backed partition metadata after routing.
    graph_clusters.counts = torch.as_tensor(graph_clusters.counts)
    cached = loss_fn(
        edge_index,
        current_pred,
        clusts=graph_clusters,
        labels=cached_inputs["target"].to_tensor(dtype=torch.float32),
        valid_mask=cached_inputs["valid"],
    )
    assert cached["count"] == 1

    with pytest.raises(TypeError, match="forest group labels"):
        loss_fn(
            edge_index,
            current_pred,
            clusts=graph_clusters,
            labels=np.zeros(3),
            valid_mask=cached_inputs["valid"],
        )
    with pytest.raises(ValueError, match="align with graph nodes"):
        loss_fn(
            edge_index,
            current_pred,
            clusts=graph_clusters,
            labels=TensorBatch(np.zeros(2), edge_index.counts),
            valid_mask=cached_inputs["valid"],
        )
    with pytest.raises(TypeError, match="validity mask must be TensorBatch"):
        loss_fn(
            edge_index,
            current_pred,
            clusts=graph_clusters,
            labels=cached_inputs["target"],
            valid_mask=np.ones(2),
        )


def test_edge_channel_high_purity(graph_labels, graph_clusters, monkeypatch):
    """High-purity edge supervision is restricted to shower groups."""
    _add_particle_fields(
        graph_labels,
        group=[0, 0, 1],
        particle=[0, 1, 0],
        group_primary=[1, 0, 1],
    )
    edge_index, edge_pred = _edge_inputs()

    with pytest.raises(ValueError, match="only valid"):
        EdgeChannelLoss(target="interaction", high_purity=True)

    monkeypatch.setattr(
        "spine.model.grappa.loss.edge_channel.edge_purity_mask_batch",
        lambda *args: np.array([True, False]),
    )
    result = EdgeChannelLoss(target="group", high_purity=True)(
        edge_index,
        edge_pred,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )
    assert result["count"] == 1


def test_node_class_weights_masks_and_metrics(graph_labels, graph_clusters):
    """Classification applies fixed and balanced weights to valid classes."""
    with pytest.raises(ValueError, match="computed on the fly"):
        NodeClassLoss(target="shape", weights=[1.0, 2.0], balance_loss=True)

    prediction = TensorBatch(
        torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
        graph_clusters.counts,
    )
    result = NodeClassLoss(target="shape", weights=[1.0, 2.0])(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )
    assert result["count"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["accuracy_class_0"] == 1.0
    assert result["accuracy_class_1"] == pytest.approx(0.5)

    result = NodeClassLoss(target="shape", balance_loss=True)(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )
    assert torch.isfinite(result["loss"])

    graph_labels.particles["shape"].data[:] = [-1, 4, -1]
    with pytest.warns(RuntimeWarning, match="larger"):
        result = NodeClassLoss(target="shape")(
            prediction,
            clust_label=graph_labels,
            clusts=graph_clusters,
        )
    assert result["count"] == 0
    assert result["accuracy"] == 1.0
    assert result["accuracy_class_0"] == 1.0


def test_node_class_closest_label_contract(
    graph_labels,
    graph_clusters,
    monkeypatch,
):
    """Closest-fragment relabeling validates and normalizes its fallback."""
    prediction = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)
    loss = NodeClassLoss(target="shape", use_closest=True)
    with pytest.raises(ValueError, match="coord_label"):
        loss(prediction, clust_label=graph_labels, clusts=graph_clusters)

    coord_label = TensorBatch(np.zeros((3, 1)), graph_clusters.counts)
    with pytest.raises(ValueError, match="exactly one"):
        NodeClassLoss(
            target="shape",
            use_closest=True,
            secondary_label=[0],
        )(
            prediction,
            clust_label=graph_labels,
            clusts=graph_clusters,
            coord_label=coord_label,
        )

    captured = []

    def closest(*args):
        captured.append(args[-1])
        return TensorBatch(np.array([0, 1, 1]), graph_clusters.counts)

    monkeypatch.setattr(
        "spine.model.grappa.loss.node_class.get_cluster_closest_label_batch",
        closest,
    )
    result = loss(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
        coord_label=coord_label,
    )
    assert result["count"] == 3
    np.testing.assert_array_equal(captured[0], [-1, -1])

    list_default_loss = NodeClassLoss(
        target="shape",
        use_closest=True,
        secondary_label=[0, 1],
    )
    result = list_default_loss(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
        coord_label=coord_label,
    )
    assert result["count"] == 3
    np.testing.assert_array_equal(captured[-1], [0, 1])


def test_shower_primary_options(graph_labels, graph_clusters, monkeypatch):
    """Shower-primary supervision supports closest and purity selections."""
    _add_particle_fields(
        graph_labels,
        group=[0, 0, 1],
        group_primary=[1, 0, 1],
    )
    prediction = TensorBatch(
        torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        graph_clusters.counts,
    )
    closest_loss = NodeShowerPrimaryLoss(use_closest=True)
    with pytest.raises(ValueError, match="coord_label"):
        closest_loss(prediction, clust_label=graph_labels, clusts=graph_clusters)

    monkeypatch.setattr(
        "spine.model.grappa.loss.node_shower_primary."
        "get_cluster_closest_primary_label_batch",
        lambda *args: TensorBatch(np.array([1, 0, 1]), graph_clusters.counts),
    )
    coord_label = TensorBatch(np.zeros((3, 1)), graph_clusters.counts)
    assert (
        closest_loss(
            prediction,
            clust_label=graph_labels,
            clusts=graph_clusters,
            coord_label=coord_label,
        )["accuracy"]
        == 1.0
    )

    with pytest.raises(ValueError, match="group predictions"):
        NodeShowerPrimaryLoss(high_purity=True, use_group_pred=True)(
            prediction,
            clust_label=graph_labels,
            clusts=graph_clusters,
        )

    monkeypatch.setattr(
        "spine.model.grappa.loss.node_shower_primary.node_purity_mask_batch",
        lambda *args: np.array([True, False, True]),
    )
    result = NodeShowerPrimaryLoss(
        balance_loss=True,
        high_purity=True,
        use_group_pred=True,
    )(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
        group_pred=TensorBatch(np.array([0, 0, 1]), graph_clusters.counts),
    )
    assert result["count"] == 2
    assert result["accuracy"] == 1.0

    graph_labels.particles["group_primary"].data.fill(-1)
    result = NodeShowerPrimaryLoss()(
        prediction, clust_label=graph_labels, clusts=graph_clusters
    )
    assert result["count"] == 0
    assert result["accuracy"] == 1.0


def test_shower_primary_reuses_cached_target(graph_labels, graph_clusters):
    """Shower-primary loss should consume its emitted static supervision."""
    _add_particle_fields(
        graph_labels,
        group=[0, 0, 1],
        group_primary=[1, 0, 1],
    )
    prediction = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)
    loss_fn = NodeShowerPrimaryLoss()
    live = loss_fn(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
        return_target=True,
    )
    cached = loss_fn(
        prediction,
        clusts=graph_clusters,
        labels=live["target"],
        valid_mask=live["valid"],
    )

    torch.testing.assert_close(cached["loss"], live["loss"])
    assert cached["count"] == live["count"]


def test_shower_primary_reapplies_predicted_group_purity(
    graph_labels,
    graph_clusters,
    monkeypatch,
):
    """Cached primary supervision should use the current predicted groups."""
    _add_particle_fields(
        graph_labels,
        group=[0, 0, 1],
        group_primary=[1, 0, 1],
    )
    prediction = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)
    monkeypatch.setattr(
        "spine.model.grappa.loss.node_shower_primary.node_purity_mask_batch",
        lambda group_ids, _: group_ids.numpy_tensor().astype(bool),
    )
    loss_fn = NodeShowerPrimaryLoss(high_purity=True, use_group_pred=True)
    initial_groups = TensorBatch(np.ones(3), graph_clusters.counts)
    current_groups = TensorBatch(np.asarray([1, 0, 0]), graph_clusters.counts)

    cached_inputs = loss_fn(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
        group_pred=initial_groups,
        return_target=True,
    )
    assert cached_inputs["count"] == 3
    np.testing.assert_array_equal(
        cached_inputs["valid"].numpy_tensor(),
        np.ones(3, dtype=bool),
    )

    live = loss_fn(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
        group_pred=current_groups,
    )
    cached = loss_fn(
        prediction,
        clusts=graph_clusters,
        group_pred=current_groups,
        labels=cached_inputs["target"],
        valid_mask=cached_inputs["valid"],
    )
    assert live["count"] == cached["count"] == 1
    torch.testing.assert_close(cached["loss"], live["loss"])

    with pytest.raises(ValueError, match="group predictions"):
        loss_fn(
            prediction,
            clusts=graph_clusters,
            labels=cached_inputs["target"],
            valid_mask=cached_inputs["valid"],
        )


def test_shower_primary_truth_group_purity(
    graph_labels,
    graph_clusters,
    monkeypatch,
):
    """High-purity selection can use truth group assignments."""
    _add_particle_fields(
        graph_labels,
        group=[0, 0, 1],
        group_primary=[1, 0, 1],
    )
    monkeypatch.setattr(
        "spine.model.grappa.loss.node_shower_primary.node_purity_mask_batch",
        lambda *args: np.ones(3, dtype=bool),
    )
    prediction = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)

    result = NodeShowerPrimaryLoss(high_purity=True)(
        prediction,
        clust_label=graph_labels,
        clusts=graph_clusters,
    )

    assert result["count"] == 3


def test_shower_primary_filters_low_quality_fragments(graph_labels):
    """Specialized shower-primary targets use the shared overlap policy."""
    _add_particle_fields(
        graph_labels,
        group=[0, 1, 0],
        group_primary=[1, 0, 1],
    )
    objects = _mixed_graph_clusters()
    prediction = TensorBatch(torch.zeros((3, 2)), objects.counts)

    result = NodeShowerPrimaryLoss(min_purity=[0.75, 0.75])(
        prediction,
        clust_label=graph_labels,
        clusts=objects,
    )

    assert result["count"] == 1
    assert result["count_rejected"] == 2


def test_node_orientation_filters_low_quality_tracks(graph_labels):
    """Track orientation retains only clusters with trustworthy associations."""
    _add_particle_fields(
        graph_labels,
        particle=[0, 1, 0],
        shape=[1, 1, 1],
    )
    objects = _mixed_graph_clusters()
    prediction = TensorBatch(torch.tensor([[0.0, 1.0]] * 3), objects.counts)
    starts = TensorBatch(torch.zeros((3, 3)), objects.counts)
    ends = TensorBatch(torch.tensor([[1.0, 0.0, 0.0]] * 3), objects.counts)
    coord_label = TensorBatch(
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 3),
        counts=[2, 1],
        coord_cols=np.arange(6),
        schema=TensorSchema(
            coordinate_groups={"start": (0, 1, 2), "end": (3, 4, 5)},
        ),
    )

    loss_fn = NodeOrientLoss(min_purity=[0.75, 0.75])
    result = loss_fn(
        prediction,
        clust_label=graph_labels,
        coord_label=coord_label,
        clusts=objects,
        start_points=starts,
        end_points=ends,
        return_target=True,
    )

    assert result["count"] == 1
    assert result["count_rejected"] == 2
    assert result["accuracy"] == 1.0

    # Cached supervision must bypass endpoint requirements at the call boundary.
    cached = loss_fn(
        prediction,
        clusts=objects,
        labels=result["target"],
        valid_mask=result["valid"],
    )
    torch.testing.assert_close(cached["loss"], result["loss"])
    assert cached["count"] == result["count"]


def test_vertex_loss_checks_containment_per_event(
    graph_labels,
    graph_clusters,
    monkeypatch,
):
    """Containment uses physical positions and masks nodes outside geometry."""

    class Geometry:
        def define_containment_volumes(self, margin, mode):
            assert margin == 0.0
            assert mode == "module"
            return "volumes"

        def check_containment(self, definition, points, summarize):
            assert definition == "volumes"
            assert not summarize
            return np.ones(len(points), dtype=bool)

    monkeypatch.setattr(
        "spine.model.grappa.loss.node_vertex.GeoManager.get_instance",
        lambda: Geometry(),
    )
    prediction = TensorBatch(torch.zeros((3, 5)), graph_clusters.counts)
    meta = [
        Meta(
            lower=np.zeros(3),
            upper=np.full(3, 20.0),
            size=np.ones(3),
            count=np.full(3, 20, dtype=np.int64),
        ),
        Meta(
            lower=np.zeros(3),
            upper=np.full(3, 20.0),
            size=np.ones(3),
            count=np.full(3, 20, dtype=np.int64),
        ),
    ]

    result = NodeVertexLoss()(graph_labels, graph_clusters, prediction, meta=meta)

    assert result["reg_count"] == 2


def test_vertex_loss_normalizes_anchor_points(graph_labels, graph_clusters):
    """Normalized anchor offsets use the scale of each batch entry."""
    prediction = TensorBatch(torch.zeros((3, 5)), graph_clusters.counts)
    starts = TensorBatch(torch.zeros((3, 3)), graph_clusters.counts)
    ends = TensorBatch(torch.ones((3, 3)), graph_clusters.counts)
    meta = [
        Meta(
            lower=np.zeros(3),
            upper=np.full(3, 10.0),
            size=np.ones(3),
            count=np.full(3, 10, dtype=np.int64),
        ),
        Meta(
            lower=np.zeros(3),
            upper=np.full(3, 20.0),
            size=np.ones(3),
            count=np.full(3, 20, dtype=np.int64),
        ),
    ]

    result = NodeVertexLoss(
        only_contained=False,
        normalize_positions=True,
        use_anchor_points=True,
    )(
        graph_labels,
        graph_clusters,
        prediction,
        meta=meta,
        start_points=starts,
        end_points=ends,
    )

    assert torch.isfinite(result["loss"])


def test_vertex_loss_shares_quality_mask_across_both_tasks(graph_labels):
    """Primary classification and vertex regression reject the same fragments."""
    objects = _mixed_graph_clusters()
    prediction = TensorBatch(torch.zeros((3, 5)), objects.counts)

    result = NodeVertexLoss(
        only_contained=False,
        min_purity=[0.75, 0.75],
    )(graph_labels, objects, prediction)

    assert result["primary_count"] == 1
    assert result["primary_count_rejected"] == 2
    assert result["reg_count"] == 1
    assert result["reg_count_rejected"] == 0


def test_edge_channel_filters_low_quality_endpoints(graph_labels):
    """Edge supervision requires both endpoint clusters to pass quality gates."""
    _add_particle_fields(graph_labels, group=[0, 1, 0])
    objects = _mixed_graph_clusters()
    edge_index, edge_pred = _edge_inputs()

    result = EdgeChannelLoss(
        target="group",
        min_purity=[0.75, 0.75],
    )(
        edge_index,
        edge_pred,
        clust_label=graph_labels,
        clusts=objects,
    )
    assert result["count"] == 0
    assert result["count_rejected"] == 2

    result = EdgeChannelLoss(
        target="group",
        mode="forest",
        min_purity=0.75,
    )(
        edge_index,
        edge_pred,
        clust_label=graph_labels,
        clusts=objects,
    )
    assert result["count"] == 0
    assert result["count_rejected"] == 2

    with pytest.raises(ValueError, match="require scalar"):
        EdgeChannelLoss(
            target="group",
            mode="forest",
            min_purity=[0.75, 0.75],
        )


def test_edge_channel_filters_cached_supervision_after_dropout():
    """Cached edge labels and validity follow the augmented graph selection."""
    edge_index = EdgeIndexBatch(
        np.array([[0, 1], [1, 0]]),
        counts=[2],
        spans=[2],
        directed=False,
    )
    edge_pred = TensorBatch(torch.tensor([[0.0, 1.0], [0.0, 1.0]]), counts=[2])
    edge_keep = TensorBatch(
        np.array([True, True, False, False]),
        counts=[4],
    )
    labels = TensorBatch(np.array([1, 1, 0, 0]), counts=[4])
    valid = TensorBatch(np.ones(4, dtype=bool), counts=[4])

    result = EdgeChannelLoss(target="group")(
        edge_index,
        edge_pred,
        labels=labels,
        valid_mask=valid,
        edge_keep=edge_keep,
        return_target=True,
    )

    assert result["count"] == 2
    np.testing.assert_array_equal(result["target"].data, [1, 1])
    np.testing.assert_array_equal(result["valid"].data, [True, True])


def test_edge_forest_dropout_preserves_node_aligned_cached_target():
    """Forest dropout filters static edge validity but not node group IDs."""
    edge_index = EdgeIndexBatch(
        np.array([[0, 1], [1, 0]]),
        counts=[2],
        spans=[2],
        directed=False,
    )
    edge_pred = TensorBatch(torch.tensor([[0.0, 1.0], [0.0, 1.0]]), counts=[2])
    edge_keep = TensorBatch(
        np.array([False, False, True, True]),
        counts=[4],
    )
    group_ids = TensorBatch(np.array([0, 0]), counts=[2])
    valid = TensorBatch(np.ones(4, dtype=bool), counts=[4])

    result = EdgeChannelLoss(target="group", mode="forest")(
        edge_index,
        edge_pred,
        labels=group_ids,
        valid_mask=valid,
        edge_keep=edge_keep,
        return_target=True,
    )

    assert result["target"] is group_ids
    assert result["valid"].counts.tolist() == [2]
