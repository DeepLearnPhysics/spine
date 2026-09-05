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


def test_grappa_loss_filters_cached_node_supervision_after_dropout():
    """The loss wrapper maps original cached node targets to retained nodes."""

    class CaptureNodeLoss(torch.nn.Module):
        def forward(self, labels=None, valid_mask=None, **kwargs):
            assert labels is not None and valid_mask is not None
            assert labels.data.tolist() == [2, 4]
            assert valid_mask.data.tolist() == [True, True]
            return {"loss": torch.tensor(0.0), "accuracy": 1.0}

    loss = GrapPALoss({"node_loss": {"name": "class", "target": "pid"}})
    loss.node_loss = CaptureNodeLoss()
    node_pred = TensorBatch(torch.zeros((2, 2)), counts=[2])

    result = loss(
        node_pred=node_pred,
        node_keep=TensorBatch(np.array([True, False, True]), counts=[3]),
        node_target=TensorBatch(np.array([2, 3, 4]), counts=[3]),
        node_valid=TensorBatch(np.ones(3, dtype=bool), counts=[3]),
    )

    assert result["accuracy"] == 1.0


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

    with pytest.raises(ValueError, match="Building edges requires both"):
        model()

    node_features = TensorBatch(torch.ones((3, 33)), graph_clusters.counts)
    with pytest.raises(ValueError, match="Building edges requires both"):
        model(node_features=node_features)

    with pytest.raises(ValueError, match="Running the node encoder requires both"):
        model(clusts=graph_clusters, edge_index=edge_index)


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


def test_grappa_forward_accepts_fully_materialized_graph():
    """Run message passing and grouping without voxel or cluster products."""
    model = GrapPA(shower_model_config())

    class MaterializedGNN(torch.nn.Module):
        node_feats = 2
        edge_feats = 1
        global_feats = 0

        def forward(self, nodes, index, edges, globals_, batch_ids):
            assert edges is not None
            assert globals_ is None
            assert index.device == nodes.device
            assert batch_ids.tolist() == [0, 0, 1]
            return {"edge_features": edges}

    class EdgeHead(torch.nn.Module):
        def forward(self, features):
            values = features.torch_tensor()
            return TensorBatch(torch.cat((-values, values), dim=1), features.counts)

    model.gnn = MaterializedGNN()
    model.node_pred_keys = []
    model.edge_pred_keys = ["edge_pred"]
    model.global_pred_keys = []
    model.edge_pred = EdgeHead()
    model.return_features = True
    model.make_groups = True
    model.grouping_method = "score"

    node_features = TensorBatch(torch.ones((3, 2)), counts=[2, 1])
    edge_features = TensorBatch(torch.ones((2, 1)), counts=[2, 0])
    edge_index = EdgeIndexBatch(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        counts=[2, 0],
        spans=[2, 1],
        directed=False,
    )

    result = model(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )

    assert "clusts" not in result
    assert result["node_features"] is node_features
    assert result["edge_features"] is edge_features
    assert result["group_pred"].counts.tolist() == [2, 1]


def test_grappa_edge_dropout_filters_materialized_graph(monkeypatch):
    """Training filters cached indexes/features while evaluation is unchanged."""
    config = shower_model_config()
    config["augment"] = {"edge_dropout": {"probability": 0.5}}
    model = GrapPA(config)

    class MaterializedGNN(torch.nn.Module):
        node_feats = 2
        edge_feats = 1
        global_feats = 0

        def forward(self, nodes, index, edges, globals_, batch_ids):
            assert edges is not None
            return {"edge_features": edges}

    model.gnn = MaterializedGNN()
    model.node_pred_keys = []
    model.edge_pred_keys = []
    model.global_pred_keys = []
    model.make_groups = False
    model.return_features = True
    node_features = TensorBatch(torch.ones((3, 2)), counts=[3])
    edge_features = TensorBatch(torch.arange(6).reshape(6, 1), counts=[6])
    edge_index = EdgeIndexBatch(
        torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
            dtype=torch.long,
        ),
        counts=[6],
        spans=[3],
        directed=False,
    )
    monkeypatch.setattr(np.random, "random", lambda _: np.array([0.1, 0.8, 0.2]))

    result = model(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )

    assert result["edge_index"].counts.tolist() == [2]
    assert result["edge_features"].data.tolist() == [[2], [3]]
    assert result["edge_keep"].data.tolist() == [False, False, True, True, False, False]

    model.eval()
    result = model(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )
    assert result["edge_index"] is edge_index
    assert result["edge_features"] is edge_features
    assert "edge_keep" not in result


def test_grappa_edge_dropout_precedes_dynamic_edge_encoding(
    graph_data,
    graph_clusters,
):
    """Dynamic edge encoders receive the augmented graph, including no edges."""
    config = shower_model_config()
    config["graph"]["max_length"] = None
    config["augment"] = {"edge_dropout": {"probability": 1.0}}
    model = GrapPA(config)

    class EdgeEncoder(torch.nn.Module):
        feature_size = model.gnn.edge_feats

        def forward(self, data, clusts, edge_index, **kwargs):
            assert edge_index.counts.tolist() == [0, 0]
            return TensorBatch(
                torch.empty((0, self.feature_size)),
                edge_index.counts,
            )

    class GNN(torch.nn.Module):
        node_feats = model.gnn.node_feats
        edge_feats = model.gnn.edge_feats
        global_feats = 0

        def forward(self, nodes, index, edges, globals_, batch_ids):
            assert index.shape == (2, 0)
            assert edges is not None and edges.shape[0] == 0
            return {}

    model.edge_encoder = EdgeEncoder()
    model.gnn = GNN()
    model.node_pred_keys = []
    model.edge_pred_keys = []
    model.global_pred_keys = []
    model.make_groups = False
    node_features = TensorBatch(
        torch.ones((3, model.gnn.node_feats)),
        graph_clusters.counts,
    )
    shapes = TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1])

    result = model(
        graph_data.to_tensor(),
        clusts=graph_clusters,
        node_features=node_features,
        shapes=shapes,
    )

    assert result["edge_index"].shape == (2, 0)


def test_grappa_augments_materialized_node_and_edge_features(monkeypatch):
    """Noise then masking operate at the common pre-GNN feature boundary."""
    config = shower_model_config()
    config["augment"] = {
        "feature_noise": {
            "node": {"sigma": 1.0, "columns": [0]},
            "edge": {"sigma": 0.5, "columns": [1], "mode": "relative"},
        },
        "feature_mask": {
            "node": {"probability": 1.0, "columns": [0]},
            "edge": {"probability": 1.0, "columns": [0]},
        },
    }
    model = GrapPA(config)

    class MaterializedGNN(torch.nn.Module):
        node_feats = 2
        edge_feats = 2
        global_feats = 0

        def forward(self, nodes, index, edges, globals_, batch_ids):
            assert edges is not None
            if model.training:
                assert nodes.data.tolist() == [[0.0, 1.0]] * 3
                assert edges.data.tolist() == [[0.0, 2.0], [0.0, 2.0]]
            return {}

    model.gnn = MaterializedGNN()
    model.node_pred_keys = []
    model.edge_pred_keys = []
    model.global_pred_keys = []
    model.make_groups = False
    model.return_features = True
    node_features = TensorBatch(torch.ones((3, 2)), counts=[2, 1])
    edge_features = TensorBatch(torch.ones((2, 2)), counts=[2, 0])
    edge_index = EdgeIndexBatch(
        torch.tensor([[0, 1], [1, 0]]),
        counts=[2, 0],
        spans=[2, 1],
        directed=False,
    )
    noise_samples = iter((np.ones((2, 1)), np.array([[2.0], [3.0]])))
    monkeypatch.setattr(np.random, "normal", lambda size: next(noise_samples))
    monkeypatch.setattr(np.random, "random", lambda shape: np.zeros(shape))

    result = model(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )

    assert result["node_features"].data.tolist() == [[0.0, 1.0]] * 3
    assert result["edge_features"].data.tolist() == [[0.0, 2.0], [0.0, 2.0]]
    assert node_features.data.tolist() == [[1.0, 1.0]] * 3

    model.eval()
    result = model(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )
    assert result["node_features"] is node_features
    assert result["edge_features"] is edge_features


def test_grappa_validates_feature_augmentation_targets_and_products():
    """Reject unknown feature families and absent configured edge features."""
    config = shower_model_config()
    config["augment"] = {"feature_mask": {"global": {"probability": 0.1}}}
    with pytest.raises(ValueError, match="target must be 'node' or 'edge'"):
        GrapPA(config)

    config = shower_model_config()
    config.pop("edge_encoder")
    config["augment"] = {
        "feature_noise": {"edge": {"sigma": 0.1}},
        "feature_mask": {"edge": {"probability": 0.1}},
    }
    model = GrapPA(config)
    model.make_groups = False
    model.node_pred_keys = []
    model.edge_pred_keys = []
    model.global_pred_keys = []
    node_features = TensorBatch(torch.ones((1, model.gnn.node_feats)), counts=[1])
    edge_index = EdgeIndexBatch(
        torch.empty((2, 0), dtype=torch.long), [0], [1], directed=False
    )

    with pytest.raises(ValueError, match="Edge feature noise requires"):
        model(node_features=node_features, edge_index=edge_index)

    model.edge_feature_noise = None
    with pytest.raises(ValueError, match="Edge feature masking requires"):
        model(node_features=node_features, edge_index=edge_index)


def test_grappa_grouped_node_dropout_filters_materialized_graph(monkeypatch):
    """Materialized node groups drive coherent node and incident-edge removal."""
    config = shower_model_config()
    config["augment"] = {
        "edge_dropout": {"probability": 0.5},
        "node_dropout": {
            "probability": 0.5,
            "group_by": "ancestor",
            "select": {"shape": ["shower", "track"]},
        },
    }
    model = GrapPA(config)

    class MaterializedGNN(torch.nn.Module):
        node_feats = 2
        edge_feats = 1
        global_feats = 0

        def forward(self, nodes, index, edges, globals_, batch_ids):
            if len(nodes.data) == 2:
                assert nodes.data.tolist() == [[0.0, 0.0], [1.0, 1.0]]
                assert index.tolist() == [[0, 1], [1, 0]]
                assert edges is not None and edges.data.tolist() == [[0.0], [1.0]]
            return {}

    model.gnn = MaterializedGNN()
    model.node_pred_keys = []
    model.edge_pred_keys = []
    model.global_pred_keys = []
    model.make_groups = False
    model.return_features = True
    node_features = TensorBatch(
        torch.arange(4, dtype=torch.float32).repeat(2, 1).T,
        counts=[4],
    )
    edge_features = TensorBatch(torch.arange(6.0).reshape(6, 1), counts=[6])
    edge_index = EdgeIndexBatch(
        torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
            dtype=torch.long,
        ),
        counts=[6],
        spans=[4],
        directed=False,
    )
    group_ids = TensorBatch(torch.tensor([0, 0, 1, 1]), counts=[4])
    eligible = TensorBatch(torch.ones(4, dtype=torch.bool), counts=[4])
    samples = iter((np.array([0.8, 0.8, 0.2]), np.array([0.8, 0.2])))
    monkeypatch.setattr(np.random, "random", lambda _: next(samples))

    result = model(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
        node_dropout_group_ids=group_ids,
        node_dropout_eligible=eligible,
    )

    assert result["node_keep"].data.tolist() == [True, True, False, False]
    assert result["edge_keep"].data.tolist() == [True, True, False, False, False, False]
    assert result["node_dropout_group_ids"].data.tolist() == [0, 0]
    assert result["node_dropout_eligible"].data.tolist() == [True, True]
    assert result["edge_index"].spans.tolist() == [2]

    model.eval()
    result = model(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )
    assert result["node_features"] is node_features
    assert "node_keep" not in result


def test_grappa_grouped_node_dropout_derives_live_labels(
    graph_labels,
    graph_clusters,
    monkeypatch,
):
    """Live structured labels supply configured physical group membership."""
    config = shower_model_config()
    config["augment"] = {
        "node_dropout": {
            "probability": 1.0,
            "group_by": "group",
            "select": {"shape": ["shower", "track"]},
        }
    }
    model = GrapPA(config)
    model.return_features = True
    model.make_groups = False
    model.node_pred_keys = []
    model.edge_pred_keys = []
    model.global_pred_keys = []

    class NodeEncoder(torch.nn.Module):
        feature_size = model.gnn.node_feats

        def forward(self, data, clusts, **kwargs):
            features = TensorBatch(
                torch.ones((3, self.feature_size)), graph_clusters.counts
            )
            points = TensorBatch(
                torch.arange(18.0).reshape(3, 6), graph_clusters.counts
            )
            return features, points

    model.node_encoder = NodeEncoder()
    edge_features = TensorBatch(torch.empty((0, model.gnn.edge_feats)), counts=[0, 0])
    edge_index = EdgeIndexBatch(
        torch.empty((2, 0), dtype=torch.long),
        counts=[0, 0],
        spans=graph_clusters.counts,
        directed=False,
    )
    monkeypatch.setattr(np.random, "randint", lambda _: 0)

    result = model(
        graph_labels,
        clusts=graph_clusters,
        edge_index=edge_index,
        edge_features=edge_features,
        shapes=TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
    )

    assert result["node_keep"].data.tolist() == [True, False, True]
    assert result["clusts"].counts.tolist() == [1, 1]
    assert result["node_dropout_group_ids"].numpy_tensor().tolist() == [0, 0]
    assert result["node_dropout_eligible"].data.tolist() == [True, True]
    assert result["start_points"].counts.tolist() == [1, 1]
    assert result["end_points"].counts.tolist() == [1, 1]

    # Feature-cache production exposes the unaugmented static routing products.
    model.eval()
    cached = model(
        graph_labels,
        clusts=graph_clusters,
        edge_index=edge_index,
        edge_features=edge_features,
        shapes=TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
    )
    assert cached["node_dropout_group_ids"].data.tolist() == [0, 1, 0]
    assert cached["node_dropout_eligible"].data.tolist() == [True, True, True]


def test_grappa_grouped_node_dropout_requires_materialized_groups():
    """Materialized grouped training cannot infer absent physical labels."""
    config = shower_model_config()
    config["augment"] = {"node_dropout": {"probability": 0.5, "group_by": "group"}}
    model = GrapPA(config)
    node_features = TensorBatch(torch.ones((1, model.gnn.node_feats)), counts=[1])
    edge_features = TensorBatch(torch.empty((0, model.gnn.edge_feats)), counts=[0])
    edge_index = EdgeIndexBatch(
        torch.empty((2, 0), dtype=torch.long), [0], [1], directed=False
    )

    with pytest.raises(ValueError, match="requires node-aligned"):
        model(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )


def test_grappa_selected_node_dropout_requires_materialized_eligibility():
    """Materialized class-conditioned training requires its cached mask."""
    config = shower_model_config()
    config["augment"] = {
        "node_dropout": {"probability": 0.5, "select": {"shape": "delta"}}
    }
    model = GrapPA(config)
    node_features = TensorBatch(torch.ones((1, model.gnn.node_feats)), counts=[1])
    edge_features = TensorBatch(torch.empty((0, model.gnn.edge_feats)), counts=[0])
    edge_index = EdgeIndexBatch(
        torch.empty((2, 0), dtype=torch.long), [0], [1], directed=False
    )

    with pytest.raises(ValueError, match="node_dropout_eligible"):
        model(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )


def test_grappa_materialized_training_forward_and_backward():
    """Train the real GrapPA network from cached graph features and targets."""
    model = GrapPA(shower_model_config())
    node_features = TensorBatch(torch.randn((3, model.gnn.node_feats)), counts=[3])
    edge_features = TensorBatch(torch.randn((6, model.gnn.edge_feats)), counts=[6])
    edge_index = EdgeIndexBatch(
        torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
            dtype=torch.long,
        ),
        counts=[6],
        spans=[3],
        directed=False,
    )

    output = model(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )
    output.update(
        node_target=TensorBatch(torch.tensor([0.0, 1.0, 0.0]), counts=[3]),
        node_valid=TensorBatch(torch.ones(3), counts=[3]),
        edge_target=TensorBatch(torch.ones(6), counts=[6]),
        edge_valid=TensorBatch(torch.ones(6), counts=[6]),
    )
    objective = GrapPALoss(
        {
            "node_loss": {"name": "class", "target": "pid"},
            "edge_loss": {"name": "channel", "target": "group"},
        }
    )

    result = objective(**output)
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_grappa_validates_materialized_graph_partitions():
    """Reject cached features whose event partitions differ from the graph."""
    model = GrapPA(shower_model_config())
    node_features = TensorBatch(torch.ones((3, model.gnn.node_feats)), counts=[1, 2])
    edge_features = TensorBatch(torch.ones((1, model.gnn.edge_feats)), counts=[0, 1])
    edge_index = EdgeIndexBatch(
        torch.tensor([[0], [1]], dtype=torch.long),
        counts=[1, 0],
        spans=[2, 1],
        directed=True,
    )

    with pytest.raises(ValueError, match="Node feature counts"):
        model(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )

    node_features = TensorBatch(torch.ones((3, model.gnn.node_feats)), counts=[2, 1])
    with pytest.raises(ValueError, match="Edge feature counts"):
        model(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )


def test_grappa_validates_materialized_feature_contracts():
    """Report missing, malformed and mispartitioned cached feature families."""
    model = GrapPA(shower_model_config())
    node_features = TensorBatch(torch.ones((3, 2)), counts=[2, 1])
    edge_features = TensorBatch(torch.ones((1, 1)), counts=[1, 0])
    edge_index = EdgeIndexBatch(
        torch.tensor([[0], [1]], dtype=torch.long),
        counts=[1, 0],
        spans=[2, 1],
        directed=True,
    )

    class SizedGNN(torch.nn.Module):
        node_feats = 2
        edge_feats = 1
        global_feats = 0

    model.gnn = SizedGNN()
    np.testing.assert_array_equal(model._counts_numpy([2, 1]), [2, 1])

    bad_nodes = TensorBatch(torch.ones((3, 3)), counts=[2, 1])
    with pytest.raises(ValueError, match="Node features contain"):
        model._validate_materialized_inputs(
            bad_nodes, edge_features, None, edge_index, None
        )

    with pytest.raises(ValueError, match="expects 1 edge features"):
        model._validate_materialized_inputs(node_features, None, None, edge_index, None)

    bad_edges = TensorBatch(torch.ones((1, 2)), counts=[1, 0])
    with pytest.raises(ValueError, match="Edge features contain"):
        model._validate_materialized_inputs(
            node_features, bad_edges, None, edge_index, None
        )

    model.gnn.global_feats = 1
    with pytest.raises(ValueError, match="expects 1 global features"):
        model._validate_materialized_inputs(
            node_features, edge_features, None, edge_index, None
        )

    wrong_batch_globals = TensorBatch(torch.ones((1, 1)), counts=[1])
    with pytest.raises(ValueError, match="same batch size"):
        model._validate_materialized_inputs(
            node_features,
            edge_features,
            wrong_batch_globals,
            edge_index,
            None,
        )

    wide_globals = TensorBatch(torch.ones((2, 2)), counts=[1, 1])
    with pytest.raises(ValueError, match="Global features contain"):
        model._validate_materialized_inputs(
            node_features, edge_features, wide_globals, edge_index, None
        )

    bad_shapes = TensorBatch(torch.zeros(3), counts=[1, 2])
    valid_globals = TensorBatch(torch.ones((2, 1)), counts=[1, 1])
    with pytest.raises(ValueError, match="Node shapes must align"):
        model._validate_materialized_inputs(
            node_features,
            edge_features,
            valid_globals,
            edge_index,
            bad_shapes,
        )


def test_grappa_materialized_path_requires_missing_encoder_inputs():
    """A configured dynamic encoder still requires data and membership."""
    model = GrapPA(shower_model_config())
    node_features = TensorBatch(torch.ones((3, model.gnn.node_feats)), counts=[2, 1])
    edge_features = TensorBatch(torch.ones((1, model.gnn.edge_feats)), counts=[1, 0])
    edge_index = EdgeIndexBatch(
        torch.tensor([[0], [1]], dtype=torch.long),
        counts=[1, 0],
        spans=[2, 1],
        directed=True,
    )

    with pytest.raises(ValueError, match="Running the edge encoder requires both"):
        model(node_features=node_features, edge_index=edge_index)

    model.global_encoder = torch.nn.Identity()
    with pytest.raises(ValueError, match="Running the global encoder requires both"):
        model(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )


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
    with pytest.raises(ValueError, match="requires cluster membership"):
        model._get_shapes(graph_labels, None)


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
    model._make_groups(result, edge_index, graph_clusters.counts)
    assert "group_pred" in result

    model.grouping_method = "score"
    model.grouping_through_track = True
    with pytest.raises(ValueError, match="provide shapes"):
        model._make_groups(result, edge_index, graph_clusters.counts)
    model._make_groups(
        result,
        edge_index,
        graph_clusters.counts,
        TensorBatch(np.array([SHOWR_SHP, TRACK_SHP, TRACK_SHP]), [2, 1]),
    )

    model.edge_pred_keys = []
    with pytest.raises(ValueError, match="provide edge predictions"):
        model._make_groups(result, edge_index, graph_clusters.counts)
    model.edge_pred_keys = ["edge_pred"]
    model.grouping_method = "invalid"
    with pytest.raises(RuntimeError, match="Unexpected grouping"):
        model._make_groups(result, edge_index, graph_clusters.counts)


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
