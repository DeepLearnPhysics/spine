"""Opt-in real-sample downstream training and truth conversion integration."""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from spine.io.collate import CollateAll
from spine.io.dataset import PIMMDataset

from .conftest import FULL_SCHEMA, round_trip_cache
from .truth_helpers import check_truth_round_trip

SAMPLE = os.environ.get("SPINE_PIMM_ROOT")
pytestmark = pytest.mark.skipif(not SAMPLE, reason="Set SPINE_PIMM_ROOT")


def test_real_truth_conversion(tmp_path):
    """Build two complete real events and preserve cm geometry and associations."""
    ds = PIMMDataset(
        data_root=SAMPLE,
        dataset_name="sim_pixel",
        schema=FULL_SCHEMA,
        dtype="float32",
        n_entry=2,
    )
    events = [ds[0], ds[1]]
    truth = check_truth_round_trip(events, tmp_path / "truth.h5")
    for index, event in enumerate(events):
        assert all(np.isnan(p.energy_deposit) for p in event["particles"])
        assert len(truth["truth_fragments"][index]) > 0
        assert len(truth["truth_particles"][index]) > 0
        assert any(len(p.children_id) for p in truth["truth_particles"][index])


@pytest.mark.parametrize("path", ["shower", "track", "inter"])
def test_real_grappa_targets_and_backward(path, tmp_path):
    """Every generic GrapPA objective has eligible targets and nonzero gradients."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from spine.config import load_config_file
    from spine.data import TensorBatch
    from spine.model.grappa import GrapPA, GrapPALoss

    torch.manual_seed(7)
    ds = PIMMDataset(
        data_root=SAMPLE,
        dataset_name="sim_pixel",
        schema=FULL_SCHEMA,
        dtype="float32",
        n_entry=2,
    )
    # Two events exercise the coordinate-table offsets used by orientation targets.
    batch = CollateAll(data_keys=ds.data_keys)([ds[0], ds[1]])
    labels = batch["clust_label"].to_tensor(dtype=torch.float32)
    coords = batch["coord_label"].to_tensor(dtype=torch.float32)
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config/full_chain/full_chain_regression.yaml"
    )
    config = load_config_file(str(config_path), download=False)["model"]["modules"]
    model_config = deepcopy(config["grappa_" + path])
    loss_config = deepcopy(config["grappa_" + path + "_loss"])
    # Keep every objective and geometry feature; shrink only the neural hidden width.
    gnn = model_config["gnn_model"]
    gnn["edge_layer"]["mlp"]["width"] = 8
    for name in ("message_mlp", "aggr_mlp"):
        gnn["node_layer"][name]["width"] = 8
    if path == "inter":
        loss_config["node_loss"]["primary"]["use_closest"] = True
    loss_config["return_targets"] = True
    model = GrapPA(model_config).train()
    objective = GrapPALoss(loss_config, model_config)

    graph = model.materialize_graph(data=labels, coord_label=coords)
    targets = objective.materialize_targets(
        clust_label=labels, coord_label=coords, **graph
    )
    for name, valid in targets.items():
        if name.endswith("_valid"):
            mask = valid.numpy_tensor().astype(bool)
            assert mask.sum() > 0, (path, name)
            values = targets[name.removesuffix("_valid") + "_target"].numpy_tensor()
            assert np.all(values[mask] >= 0), (path, name)

    # Publish fragment/particle graph targets with their real source identities.
    graph_keys = (
        "clusts",
        "edge_index",
        "node_features",
        "edge_features",
        "shapes",
        "start_points",
        "end_points",
    )
    cached = round_trip_cache(
        batch,
        {**{key: graph[key] for key in graph_keys if key in graph}, **targets},
        tmp_path / "chain.spine-cache",
        "particle_graph" if path == "inter" else "fragment_graph",
    )
    cached = {
        key: (
            value.to_tensor(dtype=torch.float32)
            if isinstance(value, TensorBatch)
            else value
        )
        for key, value in cached.items()
    }
    for name in targets:
        np.testing.assert_array_equal(
            cached[name].to_numpy().numpy_tensor(), targets[name].numpy_tensor()
        )

    # Train from the graph cache without replacing or dropping any objective.
    output = model(
        **{
            key: cached[key]
            for key in (
                "clusts",
                "edge_index",
                "node_features",
                "edge_features",
                "shapes",
            )
            if key in cached
        }
    )
    output = {**cached, **output}
    metrics = objective(clust_label=labels, coord_label=coords, **output)
    report = {}
    for name, loss in metrics.items():
        if name.endswith("_loss"):
            prefix = name.removesuffix("_loss")
            assert metrics[prefix + "_count"] > 0, (path, prefix)
            assert torch.isfinite(loss), (path, prefix)
            gradients = torch.autograd.grad(
                loss, tuple(model.parameters()), retain_graph=True, allow_unused=True
            )
            assert all(torch.isfinite(g).all() for g in gradients if g is not None)
            assert any(g is not None and g.abs().sum() > 0 for g in gradients), (
                path,
                prefix,
            )
            report[prefix] = {
                "count": metrics[prefix + "_count"],
                "loss": float(loss.detach()),
            }
    metrics["loss"].backward()
    print({"path": path, "objectives": report})
