"""Behavioral contracts for the top-level GrapPA model and objective."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from spine.config import load_config_file
from spine.constants import DELTA_SHP, MICHL_SHP, SHOWR_SHP, TRACK_SHP
from spine.data import ClusterLabelBatch, EdgeIndexBatch, TensorBatch
from spine.model.grappa import GrapPA, GrapPALoss
from spine.model.grappa.graph import CompleteGraph

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def shower_model_config():
    """Return an independent copy of the maintained shower model block."""
    cfg = load_config_file(
        str(CONFIG_DIR / "grappa_shower" / "grappa_shower_train.yaml"),
        download=False,
    )
    return deepcopy(cfg["model"]["modules"]["grappa"])


def test_grappa_unpacks_dbscan_configuration():
    """Forward DBSCAN options by keyword while sharing node parameters."""
    grappa_cfg = shower_model_config()
    grappa_cfg["dbscan"] = {"break_shapes": []}

    model = GrapPA(grappa_cfg)

    assert model.dbscan is not None
    assert model.dbscan.shapes == model.node_type


def test_grappa_constructs_global_encoder() -> None:
    """The optional global encoder is discovered and width-validated."""
    config = shower_model_config()
    config["global_encoder"] = {"name": "empty"}
    model = GrapPA(config)
    assert model.global_encoder is not None
    assert model.global_encoder.feature_size == 0


def test_grappa_accepts_named_node_source_and_integer_shapes():
    """Accept named data fields and canonical integer semantic shapes."""
    grappa_cfg = shower_model_config()
    grappa_cfg["nodes"]["source"] = "cluster"
    grappa_cfg["nodes"]["shapes"] = [SHOWR_SHP, MICHL_SHP, DELTA_SHP]

    model = GrapPA(grappa_cfg)

    assert model.node_source == "cluster"
    assert model.node_type == [SHOWR_SHP, MICHL_SHP, DELTA_SHP]


def test_grappa_rejects_encoder_gnn_width_mismatch():
    """Report incompatible feature widths before the first forward pass."""
    grappa_cfg = shower_model_config()
    grappa_cfg["gnn_model"]["node_feats"] += 1

    with pytest.raises(ValueError, match="node encoder produces 33 features"):
        GrapPA(grappa_cfg)


def test_grappa_infers_shapes_for_track_restricted_grouping(
    graph_labels,
    graph_clusters,
):
    """Infer node shapes when track-restricted grouping requires them."""
    grappa_cfg = shower_model_config()
    grappa_cfg["nodes"]["grouping_through_track"] = True
    grappa_cfg["graph"]["max_length"] = 100.0
    model = GrapPA(grappa_cfg)

    shapes = model._get_shapes(graph_labels, graph_clusters)

    assert shapes is not None
    assert len(shapes.numpy_tensor()) == len(graph_clusters.index_list)


def test_grappa_normalizes_explicit_cached_shapes(graph_data, graph_clusters):
    """Cached semantic classes recover an integer dtype before graph indexing."""
    model = GrapPA(shower_model_config())
    cached_shapes = TensorBatch(
        torch.tensor([SHOWR_SHP, TRACK_SHP, TRACK_SHP], dtype=torch.float32),
        [2, 1],
    )

    shapes = model._get_shapes(graph_data, graph_clusters, cached_shapes)

    assert shapes is not None
    assert shapes.numpy_tensor().dtype == np.int64
    edge_index, _ = model._make_edge_index(graph_data, graph_clusters, shapes)
    assert edge_index.counts.tolist() == [0, 0]


def test_grappa_loss_routes_graph_truth_to_edge_objective():
    """Expose parsed graph labels under the edge-loss interface name."""

    class CaptureEdgeLoss(torch.nn.Module):
        """Record the true graph received from the GrapPA loss wrapper."""

        def __init__(self):
            super().__init__()
            self.true_edge_index = None

        def forward(self, true_edge_index=None, **kwargs):
            self.true_edge_index = true_edge_index
            return {
                "loss": torch.tensor(0.0),
                "accuracy": 1.0,
            }

    loss = GrapPALoss(
        {
            "edge_loss": {
                "name": "channel",
                "target": "group",
            }
        }
    )
    capture = CaptureEdgeLoss()
    loss.edge_loss = capture

    graph_label = EdgeIndexBatch(
        torch.empty((2, 0), dtype=torch.long),
        counts=[0],
        spans=[0],
        directed=True,
    )
    clust_label = ClusterLabelBatch(
        TensorBatch(
            torch.empty((0, 7)),
            counts=[0],
            has_batch_col=True,
            coord_cols=np.arange(1, 4),
        ),
        {"group": TensorBatch(torch.empty(0, dtype=torch.long), counts=[0])},
    )
    edge_pred = TensorBatch(torch.empty((0, 2)), counts=[0])

    result = loss(
        clust_label=clust_label,
        graph_label=graph_label,
        edge_pred=edge_pred,
    )

    assert capture.true_edge_index is graph_label
    assert result["accuracy"] == 1.0


def test_grappa_loss_shares_one_overlap_cache_across_objectives(graph_labels):
    """Every objective in one forward pass should receive the same cache."""

    class CaptureLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = None

        def forward(self, overlap_cache=None, **kwargs):
            self.cache = overlap_cache
            return {"loss": torch.tensor(0.0), "accuracy": 1.0}

    loss = GrapPALoss(
        {
            "node_loss": {
                "first": {"name": "class", "target": "shape"},
                "second": {"name": "class", "target": "shape"},
            }
        }
    )
    first = CaptureLoss()
    second = CaptureLoss()
    loss.node_first_loss = first
    loss.node_second_loss = second
    prediction = TensorBatch(torch.zeros((3, 2)), counts=[2, 1])

    loss(
        graph_labels,
        node_first_pred=prediction,
        node_second_pred=prediction,
    )

    assert first.cache is second.cache
    assert first.cache == {}


def test_grappa_loss_returns_and_reuses_cached_targets(
    graph_labels,
    graph_clusters,
):
    """Opt-in targets should replay node and edge supervision without truth."""
    edge_index = EdgeIndexBatch(
        np.array([[0], [1]], dtype=np.int64),
        counts=[1, 0],
        spans=graph_clusters.counts,
        directed=True,
    )
    node_pred = TensorBatch(torch.zeros((3, 2)), graph_clusters.counts)
    edge_pred = TensorBatch(torch.zeros((1, 2)), counts=[1, 0])
    cfg = {
        "node_loss": {"name": "class", "target": "pid"},
        "edge_loss": {"name": "channel", "target": "group"},
        "return_targets": True,
    }

    live = GrapPALoss(cfg)(
        graph_labels,
        clusts=graph_clusters,
        edge_index=edge_index,
        node_pred=node_pred,
        edge_pred=edge_pred,
    )

    assert live["node_target"].counts.tolist() == [2, 1]
    assert live["node_valid"].numpy_tensor().tolist() == [True, True, True]
    assert live["edge_target"].counts.tolist() == [1, 0]
    assert live["edge_valid"].numpy_tensor().tolist() == [True]

    cached_cfg = dict(cfg)
    cached_cfg["return_targets"] = False
    cached = GrapPALoss(cached_cfg)(
        clust_label=None,
        clusts=graph_clusters,
        edge_index=edge_index,
        node_pred=node_pred,
        edge_pred=edge_pred,
        node_target=live["node_target"].to_tensor(dtype=torch.float32),
        node_valid=live["node_valid"].to_tensor(dtype=torch.float32),
        edge_target=live["edge_target"].to_tensor(dtype=torch.float32),
        edge_valid=live["edge_valid"].to_tensor(dtype=torch.float32),
    )

    torch.testing.assert_close(cached["loss"], live["loss"])
    assert cached["accuracy"] == live["accuracy"]


def test_grappa_loss_routes_cached_forest_primitives(
    graph_labels,
    graph_clusters,
):
    """Forest caching may pair node targets with edge validity safely."""
    graph_labels.particles["group"] = TensorBatch(
        np.asarray([0, 0, 1]),
        graph_clusters.counts,
    )
    edge_index = EdgeIndexBatch(
        np.array([[0, 1], [1, 0]], dtype=np.int64),
        counts=[2, 0],
        spans=graph_clusters.counts,
        directed=True,
    )
    edge_pred = TensorBatch(torch.zeros((2, 2)), counts=edge_index.counts)
    cfg = {
        "edge_loss": {"name": "channel", "target": "group", "mode": "forest"},
        "return_targets": True,
    }

    live = GrapPALoss(cfg)(
        graph_labels,
        clusts=graph_clusters,
        edge_index=edge_index,
        edge_pred=edge_pred,
    )
    assert live["edge_target"].counts.tolist() == [2, 1]
    assert live["edge_valid"].counts.tolist() == [2, 0]

    cached = GrapPALoss({**cfg, "return_targets": False})(
        clust_label=None,
        clusts=graph_clusters,
        edge_index=edge_index,
        edge_pred=edge_pred,
        edge_target=live["edge_target"].to_tensor(dtype=torch.float32),
        edge_valid=live["edge_valid"].to_tensor(dtype=torch.float32),
    )
    torch.testing.assert_close(cached["loss"], live["loss"])
    assert cached["accuracy"] == live["accuracy"]


def test_grappa_loss_validates_cached_target_contract(graph_labels):
    """Cached supervision must be paired and supported by the objective."""
    prediction = TensorBatch(torch.zeros((3, 2)), counts=[2, 1])
    loss = GrapPALoss({"node_loss": {"name": "class", "target": "pid"}})
    with pytest.raises(ValueError, match="requires both"):
        loss(graph_labels, node_pred=prediction, node_target=prediction)

    with pytest.raises(ValueError, match="does not support"):
        GrapPALoss(
            {
                "node_loss": {"name": "vertex", "only_contained": False},
                "return_targets": True,
            }
        )(graph_labels, node_pred=TensorBatch(torch.zeros((3, 5)), counts=[2, 1]))

    class MissingTargetLoss(torch.nn.Module):
        def forward(self, **kwargs):
            return {"loss": torch.tensor(0.0), "accuracy": 1.0}

    loss = GrapPALoss(
        {
            "node_loss": {"name": "class", "target": "pid"},
            "return_targets": True,
        }
    )
    loss.node_loss = MissingTargetLoss()
    with pytest.raises(RuntimeError, match="did not return"):
        loss(graph_labels, node_pred=prediction)


def test_grappa_validates_node_graph_and_grouping_configuration():
    """GrapPA rejects ambiguous node construction and grouping settings."""
    cfg = shower_model_config()
    cfg.pop("nodes")
    with pytest.raises(ValueError, match="provide a `nodes`"):
        GrapPA(cfg)

    cfg = shower_model_config()
    cfg["nodes"]["shapes"] = TRACK_SHP
    with pytest.raises(ValueError, match="provided as a list"):
        GrapPA(cfg)

    cfg = shower_model_config()
    cfg["nodes"]["grouping_method"] = "mystery"
    with pytest.raises(ValueError, match="Grouping method"):
        GrapPA(cfg)

    cfg = shower_model_config()
    cfg["nodes"].update(
        grouping_method="threshold",
        grouping_through_track=True,
    )
    with pytest.raises(ValueError, match="only supported.*score"):
        GrapPA(cfg)

    cfg = shower_model_config()
    cfg["nodes"].pop("shapes")
    model = GrapPA(cfg)
    assert model.node_type == list(range(4))

    cfg = shower_model_config()
    cfg["nodes"].update(source="voxel", min_size=2)
    with pytest.raises(ValueError, match="Voxel nodes are singletons"):
        GrapPA(cfg)


def test_grappa_voxel_nodes_filter_structured_labels(graph_labels):
    """Voxel node construction should apply semantic filters to truth products."""
    cfg = shower_model_config()
    cfg["nodes"].update(source="voxel", min_size=1, shapes=[SHOWR_SHP])
    cfg["graph"].update(max_length=None, dist_algorithm="brute")
    model = GrapPA(cfg)
    clusters = model._make_clusters(graph_labels)
    assert all(len(cluster) == 1 for cluster in clusters.index_list)


def test_grappa_validates_dbscan_and_group_prediction_configuration():
    """Shared DBSCAN fields and group construction require one owner/head."""
    cfg = shower_model_config()
    cfg["dbscan"] = {"shapes": [SHOWR_SHP]}
    with pytest.raises(ValueError, match="Do not specify"):
        GrapPA(cfg)

    cfg = shower_model_config()
    cfg["nodes"]["make_groups"] = True
    cfg["gnn_model"]["edge_pred"] = None
    with pytest.raises(ValueError, match="requires an edge prediction"):
        GrapPA(cfg)


def test_grappa_forward_validates_required_explicit_inputs(
    graph_data,
    graph_clusters,
):
    """Tensor inputs require explicit graph products or configured builders."""
    model = GrapPA(shower_model_config())
    data = graph_data.to_tensor()
    with pytest.raises(TypeError, match="structured cluster labels"):
        model(data)

    model.graph_constructor = None
    with pytest.raises(ValueError, match="edge_index or graph configuration"):
        model(data, clusts=graph_clusters)
    with pytest.raises(ValueError, match="graph configuration"):
        model._make_edge_index(data, graph_clusters)

    edge_index, _, _ = CompleteGraph(directed=True)(data, graph_clusters)
    model.node_encoder = None
    with pytest.raises(ValueError, match="node_features or node encoder"):
        model(data, clusts=graph_clusters, edge_index=edge_index)


def test_grappa_forward_routes_encoded_features(
    graph_data,
    graph_clusters,
):
    """Node, edge and global encoders publish features to the GNN contract."""
    model = GrapPA(shower_model_config())
    edge_index, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)

    class NodeEncoder(torch.nn.Module):
        feature_size = 2

        def forward(self, data, clusts, **kwargs):
            return TensorBatch(torch.ones((3, 2)), clusts.counts)

    class EdgeEncoder(torch.nn.Module):
        feature_size = 2

        def forward(self, data, clusts, edge_index, **kwargs):
            return TensorBatch(torch.ones((1, 2)), edge_index.counts)

    class GlobalEncoder(torch.nn.Module):
        feature_size = 2

        def forward(self, data, clusts):
            return TensorBatch(torch.ones((2, 2)), counts=[1, 1])

    class GNN(torch.nn.Module):
        def forward(self, nodes, edges, edge_features, global_features, batch):
            assert edge_features is not None
            assert global_features is not None
            return {}

    model.node_encoder = NodeEncoder()
    model.edge_encoder = EdgeEncoder()
    model.global_encoder = GlobalEncoder()
    model.gnn = GNN()
    model.node_pred_keys = []
    model.edge_pred_keys = []
    model.global_pred_keys = []
    model.return_features = True
    model.make_groups = False

    result = model(
        graph_data.to_tensor(),
        clusts=graph_clusters,
        edge_index=edge_index,
        shapes=TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
    )

    assert result["node_features"].shape == (3, 2)
    assert result["edge_features"].shape == (1, 2)
    assert result["global_features"].shape == (2, 2)


def test_grappa_rejects_missing_gnn_prediction_features(
    graph_data, graph_clusters
) -> None:
    """Prediction heads fail clearly when the GNN omits their feature family."""
    model = GrapPA(shower_model_config())
    edge_index, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)

    class EmptyGNN(torch.nn.Module):
        def forward(self, *_args):
            return {}

    model.gnn = EmptyGNN()
    model.make_groups = False
    with pytest.raises(RuntimeError, match="did not produce `node_features`"):
        model(
            graph_data.to_tensor(),
            clusts=graph_clusters,
            edge_index=edge_index,
            node_features=TensorBatch(torch.ones((3, 33)), graph_clusters.counts),
            shapes=TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
        )


def test_grappa_dbscan_cluster_path_and_shape_contract(
    graph_labels, graph_data, graph_clusters
) -> None:
    """DBSCAN node construction consumes structured shapes; plain data cannot."""
    model = GrapPA(shower_model_config())

    class FakeDBSCAN(torch.nn.Module):
        def forward(self, data, seg_label, coord_label):
            np.testing.assert_array_equal(
                seg_label.numpy_tensor(), graph_labels.shapes.numpy_tensor()
            )
            return graph_clusters, object()

    model.dbscan = FakeDBSCAN()
    assert model._make_clusters(graph_labels) is graph_clusters

    with pytest.raises(TypeError, match="structured labels"):
        model._make_clusters(graph_data.to_tensor())

    model.graph_constructor.max_length = np.ones((5, 5))
    with pytest.raises(TypeError, match="structured cluster labels"):
        model._get_shapes(graph_data.to_tensor(), graph_clusters)


def test_grappa_endpoint_encoder_contract(
    graph_data,
    graph_clusters,
):
    """Endpoint-producing node encoders return exactly two 3D points."""
    model = GrapPA(shower_model_config())
    edge_index, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)
    features = TensorBatch(torch.ones((3, 2)), graph_clusters.counts)

    class Encoder(torch.nn.Module):
        feature_size = 2

        def __init__(self, points):
            super().__init__()
            self.points = points

        def forward(self, *args, **kwargs):
            return features, self.points

    class GNN(torch.nn.Module):
        def forward(self, *args):
            return {}

    model.gnn = GNN()
    model.node_pred_keys = []
    model.edge_pred_keys = []
    model.global_pred_keys = []
    model.edge_encoder = None
    model.make_groups = False

    model.node_encoder = Encoder(
        TensorBatch(torch.zeros((3, 6)), graph_clusters.counts)
    )
    result = model(
        graph_data.to_tensor(),
        clusts=graph_clusters,
        edge_index=edge_index,
        shapes=TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
    )
    assert result["start_points"].shape == (3, 3)
    assert result["end_points"].shape == (3, 3)

    model.node_encoder = Encoder(
        TensorBatch(torch.zeros((3, 5)), graph_clusters.counts)
    )
    with pytest.raises(ValueError, match="six coordinates"):
        model(
            graph_data.to_tensor(),
            clusts=graph_clusters,
            edge_index=edge_index,
            shapes=TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
        )

    class Malformed(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return features, features, features

    model.node_encoder = Malformed()
    with pytest.raises(TypeError, match="pair of TensorBatch"):
        model(
            graph_data.to_tensor(),
            clusts=graph_clusters,
            edge_index=edge_index,
            shapes=TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
        )


def test_grappa_group_construction_modes(graph_data, graph_clusters):
    """Threshold, track-restricted and invalid grouping modes are explicit."""
    model = GrapPA(shower_model_config())
    edge_index, _, _ = CompleteGraph(directed=True)(graph_data, graph_clusters)
    result = {"edge_pred": TensorBatch(torch.tensor([[0.0, 2.0]]), [1, 0])}
    model.edge_pred_keys = ["edge_pred"]

    model.grouping_method = "threshold"
    model._make_groups(result, graph_clusters, edge_index)
    assert "group_pred" in result

    model.grouping_method = "score"
    model.grouping_through_track = True
    with pytest.raises(ValueError, match="provide shapes"):
        model._make_groups(result, graph_clusters, edge_index)
    model._make_groups(
        result,
        graph_clusters,
        edge_index,
        TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
    )

    model.edge_pred_keys = []
    with pytest.raises(ValueError, match="provide edge predictions"):
        model._make_groups(result, graph_clusters, edge_index)
    model.edge_pred_keys = ["edge_pred"]
    model.grouping_method = "invalid"
    with pytest.raises(RuntimeError, match="Unexpected grouping"):
        model._make_groups(result, graph_clusters, edge_index)


def test_grappa_loss_validates_configuration_outputs_and_return_type(
    graph_labels,
):
    """GrapPA loss requires objectives, corresponding logits and tensor losses."""
    with pytest.raises(ValueError, match="at least one"):
        GrapPALoss({})

    loss = GrapPALoss({"node_loss": {"name": "class", "target": "shape"}})
    with pytest.raises(KeyError, match="node_pred"):
        loss(graph_labels)

    class BadLoss(torch.nn.Module):
        def forward(self, **kwargs):
            return {"loss": 1.0, "accuracy": 1.0}

    loss.node_loss = BadLoss()
    with pytest.raises(TypeError, match="torch.Tensor"):
        loss(
            graph_labels,
            node_pred=TensorBatch(torch.zeros((3, 2)), [2, 1]),
        )
