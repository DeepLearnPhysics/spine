"""Behavioral contracts for the top-level GrapPA model and objective."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from spine.config import load_config_file
from spine.constants import CLUST_COL, DELTA_SHP, MICHL_SHP, SHOWR_SHP
from spine.data import EdgeIndexBatch, TensorBatch
from spine.model.grappa import GrapPA, GrapPALoss

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


def test_grappa_accepts_integer_node_enums():
    """Accept canonical integer columns and shapes as documented."""
    grappa_cfg = shower_model_config()
    grappa_cfg["nodes"]["source"] = CLUST_COL
    grappa_cfg["nodes"]["shapes"] = [SHOWR_SHP, MICHL_SHP, DELTA_SHP]

    model = GrapPA(grappa_cfg)

    assert model.node_source == CLUST_COL
    assert model.node_type == [SHOWR_SHP, MICHL_SHP, DELTA_SHP]


def test_grappa_rejects_encoder_gnn_width_mismatch():
    """Report incompatible feature widths before the first forward pass."""
    grappa_cfg = shower_model_config()
    grappa_cfg["gnn_model"]["node_feats"] += 1

    with pytest.raises(ValueError, match="node encoder produces 33 features"):
        GrapPA(grappa_cfg)


def test_grappa_infers_shapes_for_track_restricted_grouping(
    graph_data,
    graph_clusters,
):
    """Infer node shapes when track-restricted grouping requires them."""
    grappa_cfg = shower_model_config()
    grappa_cfg["nodes"]["grouping_through_track"] = True
    grappa_cfg["graph"]["max_length"] = 100.0
    model = GrapPA(grappa_cfg)

    shapes = model._get_shapes(graph_data, graph_clusters)

    assert shapes is not None
    assert len(shapes.numpy_tensor()) == len(graph_clusters.index_list)


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
    clust_label = TensorBatch(torch.empty((0, 1)), counts=[0])
    edge_pred = TensorBatch(torch.empty((0, 2)), counts=[0])

    result = loss(
        clust_label=clust_label,
        graph_label=graph_label,
        edge_pred=edge_pred,
    )

    assert capture.true_edge_index is graph_label
    assert result["accuracy"] == 1.0
