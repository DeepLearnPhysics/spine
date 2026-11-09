"""Opt-in real-file training smoke test using models available on main.

Run with SPINE_PIMM_ROOT pointing at paired hits/labl directories.
"""

import os

import numpy as np
import pytest

from .conftest import round_trip_cache

SAMPLE = os.environ.get("SPINE_PIMM_ROOT")
pytestmark = pytest.mark.skipif(
    not SAMPLE, reason="Set SPINE_PIMM_ROOT for real-file training"
)


def test_real_pimm_forward_backward(tmp_path):
    """Backpropagate every training objective through the complete real event."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("MinkowskiEngine")
    pytest.importorskip("torch_cluster")
    from spine.io.collate import CollateAll
    from spine.io.dataset import PIMMDataset
    from spine.model.full_chain.providers.segmentation import SegmentationStage
    from spine.model.full_chain.state import ChainState
    from spine.model.graph_spice import GraphSPICE, GraphSPICELoss
    from spine.model.uresnet.ppn import UResNetPPN, UResNetPPNLoss

    torch.manual_seed(7)
    schema = {
        key: {"provider": provider}
        for key, provider in (
            ("data", "tensor"),
            ("seg_label", "segmentation"),
            ("clust_label", "cluster_label"),
            ("ppn_label", "particle_points"),
        )
    }
    ds = PIMMDataset(
        data_root=SAMPLE,
        dataset_name="sim_pixel",
        dtype="float32",
        schema=schema,
        n_entry=1,
    )
    e = ds[0]
    np.testing.assert_array_equal(
        e["data"].coordinate_data, e["seg_label"].coordinate_data
    )
    np.testing.assert_array_equal(e["data"].coordinate_data, e["clust_label"].coords)
    np.testing.assert_array_equal(
        e["seg_label"].features[:, 0], e["clust_label"].shapes
    )
    # Keep the full real event, including all overlaps and endpoint targets.
    source_batch = CollateAll(data_keys=ds.data_keys)([e])
    batch = CollateAll(data_keys=schema)([e])
    batch = {k: v.to_tensor(dtype=torch.float32) for k, v in batch.items()}
    size = int(e["data"].meta.count.max()) + 2
    backbone = {
        "num_classes": 5,
        "num_input": 1,
        "filters": 4,
        "depth": 2,
        "reps": 1,
        "spatial_size": size,
        "activation": "relu",
        "norm_layer": "none",
    }

    # Segmentation and PPN share the standard UResNet backbone on main.
    model = UResNetPPN(backbone, ppn={}).train()
    objective = UResNetPPNLoss(
        backbone, uresnet_loss={}, ppn={}, ppn_loss={"return_mask_labels": True}
    )
    stage = SegmentationStage("segmentation", "uresnet", model, None)
    output = stage(ChainState(**batch)).outputs
    metrics = objective(batch["seg_label"], batch["ppn_label"], **output)
    cached = round_trip_cache(
        source_batch,
        {"seg_pred": output["seg_pred"], "ppn_points": output["ppn_points"]},
        tmp_path / "segmentation.spine-cache",
        "segmentation",
    )
    for name in cached:
        np.testing.assert_allclose(
            cached[name].to_numpy().numpy_tensor(),
            output[name].to_numpy().numpy_tensor(),
        )
    assert len(e["seg_label"].features) > 0
    assert len(e["ppn_label"].features) > 0
    positive_points = int(metrics["ppn_mask_labels"][0].torch_tensor().sum())
    assert positive_points > 0
    assert metrics["ppn_type_loss"] > 0 and metrics["ppn_reg_loss"] > 0
    parameter = next(model.uresnet.backbone.parameters())
    report = {}
    for name in ("uresnet_loss", "ppn_loss"):
        assert torch.isfinite(metrics[name])
        grad = torch.autograd.grad(metrics[name], parameter, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        report[name] = float(metrics[name].detach())

    # Graph-SPICE uses the same aligned voxel labels with its own mainline backbone.
    graph_config = {
        "use_raw_features": True,
        "embedder": {
            "coord_conv": False,
            "uresnet": {
                key: value for key, value in backbone.items() if key != "num_classes"
            },
        },
        "kernel": {"provider": "bilinear", "num_features": 4},
        "constructor": {
            "graph": {"provider": "radius", "r": 1.9},
            "min_size": 1,
            "edge_threshold": 0.5,
        },
    }
    graph = GraphSPICE(graph_config).train()
    graph_objective = GraphSPICELoss(graph_config, {"provider": "edge", "metric": None})
    graph_metrics = graph_objective(
        batch["seg_label"],
        batch["clust_label"],
        **graph(batch["data"], batch["seg_label"]),
    )
    assert graph_metrics["count"] > 0
    assert torch.isfinite(graph_metrics["loss"])
    graph_parameter = next(graph.embedder.backbone.parameters())
    grad = torch.autograd.grad(
        graph_metrics["loss"], graph_parameter, retain_graph=True
    )[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    report["graph_spice_loss"] = float(graph_metrics["loss"].detach())

    total = metrics["loss"] + graph_metrics["loss"]
    total.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in graph.kernel_fn.parameters()
    )
    print(
        {
            "voxels": len(e["data"].features),
            "ppn_targets": len(e["ppn_label"].features),
            "ppn_positive_voxels": positive_points,
            "graph_spice_edges": graph_metrics["count"],
            "losses": report,
            "total_loss": float(total.detach()),
        }
    )
