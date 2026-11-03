"""Opt-in real-file training smoke test using models available on main.

Run with SPINE_SUPERA_SAMPLE pointing at a pysupera 4.1 production file.
"""

import os

import numpy as np
import pytest

SAMPLE = os.environ.get("SPINE_SUPERA_SAMPLE")
pytestmark = pytest.mark.skipif(
    not SAMPLE, reason="Set SPINE_SUPERA_SAMPLE for real-file training"
)


def test_real_supera_forward_backward():
    """Backpropagate every training objective through the complete real event."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("MinkowskiEngine")
    pytest.importorskip("torch_cluster")
    from spine.io.collate import CollateAll
    from spine.io.dataset import SuperaDataset
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
    ds = SuperaDataset(file_keys=SAMPLE, dtype="float32", schema=schema, n_entry=1)
    e = ds[0]
    np.testing.assert_array_equal(
        e["data"].coordinate_data, e["seg_label"].coordinate_data
    )
    np.testing.assert_array_equal(e["data"].coordinate_data, e["clust_label"].coords)
    np.testing.assert_array_equal(
        e["seg_label"].features[:, 0], e["clust_label"].shapes
    )
    # Keep the full real event, including all overlaps and endpoint targets.
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
    objective = UResNetPPNLoss(backbone, uresnet_loss={}, ppn={}, ppn_loss={})
    metrics = objective(batch["seg_label"], batch["ppn_label"], **model(batch["data"]))
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
            "losses": report,
            "total_loss": float(total.detach()),
        }
    )
