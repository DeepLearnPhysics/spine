"""Tests for strict model-checkpoint composition."""

from __future__ import annotations

import pytest

from spine.model import checkpoint_sha256, export_model_weights, verify_checkpoint
from spine.model.checkpoint import save_checkpoint
from spine.model.manager import ModelManager
from spine.utils.conditional import TORCH_AVAILABLE, torch

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required."),
]


class CompositeNetwork(torch.nn.Module):
    """Small network whose modules mimic full-chain checkpoint namespaces."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)


def make_config(encoder_path, head_path=None):
    """Build a minimal configuration with standalone component weights."""
    modules = {
        "encoder": {
            "weight_path": str(encoder_path),
            "metadata": [{"label": "component", "weight_path": "discard.ckpt"}],
        }
    }
    if head_path is not None:
        modules["head"] = {"weight_path": str(head_path)}
    else:
        modules["head"] = {}
    return {
        "base": {"world_size": 2, "epochs": 3},
        "metadata": [{"label": "composed", "weight_path": "preserve.ckpt"}],
        "train": {"save_step": 1},
        "io": {"loader": {"dataset": {"name": "test"}, "shuffle": True}},
        "model": {
            "name": "test",
            "network_input": {"data": "data"},
            "loss_input": {"target": "target"},
            "modules": modules,
        },
    }


def save_component(path, module):
    """Save one standalone module checkpoint and return its state."""
    state = module.state_dict()
    save_checkpoint({"state_dict": state}, path)
    return state


def test_export_model_weights_composes_standalone_checkpoints(monkeypatch, tmp_path):
    """Composition should remap, verify and describe all component weights."""
    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (CompositeNetwork, None),
    )
    encoder_path = tmp_path / "encoder.ckpt"
    head_path = tmp_path / "head.ckpt"
    encoder_state = save_component(encoder_path, torch.nn.Linear(2, 2))
    head_state = save_component(head_path, torch.nn.Linear(2, 1))
    output_path = tmp_path / "composite.ckpt"

    digest = export_model_weights(
        make_config(encoder_path, head_path),
        output_path,
    )
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)

    assert verify_checkpoint(output_path)
    assert digest == checkpoint_sha256(output_path)
    for key, value in encoder_state.items():
        assert torch.equal(checkpoint["state_dict"][f"encoder.{key}"], value)
    for key, value in head_state.items():
        assert torch.equal(checkpoint["state_dict"][f"head.{key}"], value)

    assert "train" not in checkpoint["config"]
    assert checkpoint["config"]["base"]["world_size"] == 0
    assert checkpoint["config"]["io"]["loader"]["shuffle"] is False
    assert checkpoint["config"]["metadata"] == [
        {"label": "composed", "weight_path": "preserve.ckpt"}
    ]
    assert checkpoint["config"]["model"]["modules"]["encoder"]["metadata"] == [
        {"label": "component"}
    ]
    assert [source["module"] for source in checkpoint["weight_sources"]] == [
        "head",
        "encoder",
    ]
    assert checkpoint["weight_sources"][0]["sha256"] == checkpoint_sha256(head_path)
    assert "optimizer" not in checkpoint
    assert "global_step" not in checkpoint

    # The inference-only artifact remains a first-class global SPINE weight
    # file despite intentionally carrying no training progress fields.
    reloaded = ModelManager(
        name="test",
        modules={"encoder": {}, "head": {}},
        network_input={"data": "data"},
        weight_path=str(output_path),
    )
    assert reloaded.start_iteration == 0
    for key, value in checkpoint["state_dict"].items():
        assert torch.equal(reloaded.net.state_dict()[key], value)


def test_export_model_weights_rejects_unpopulated_state(monkeypatch, tmp_path):
    """Constructor-initialized state must never leak into an export."""
    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (CompositeNetwork, None),
    )
    encoder_path = tmp_path / "encoder.ckpt"
    save_component(encoder_path, torch.nn.Linear(2, 2))

    with pytest.raises(ValueError, match="head.bias"):
        export_model_weights(
            make_config(encoder_path),
            tmp_path / "incomplete.ckpt",
        )


def test_export_model_weights_validates_inputs(monkeypatch, tmp_path):
    """Composition requires model configuration and distinct source/output paths."""
    with pytest.raises(KeyError, match="model"):
        export_model_weights({"io": {}}, tmp_path / "output.ckpt")
    with pytest.raises(TypeError, match="model.*mapping"):
        export_model_weights(
            {"io": {}, "model": "invalid"},
            tmp_path / "output.ckpt",
        )
    with pytest.raises(ValueError, match="weight_list"):
        export_model_weights(
            {"io": {}, "model": {"weight_list": "weights.txt"}},
            tmp_path / "output.ckpt",
        )

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (CompositeNetwork, None),
    )
    encoder_path = tmp_path / "encoder.ckpt"
    head_path = tmp_path / "head.ckpt"
    save_component(encoder_path, torch.nn.Linear(2, 2))
    save_component(head_path, torch.nn.Linear(2, 1))
    cfg = make_config(encoder_path, head_path)
    with pytest.raises(ValueError, match="cannot overwrite"):
        export_model_weights(cfg, encoder_path)


def test_export_model_weights_requires_sources_and_reports_large_gaps(
    monkeypatch, tmp_path
):
    """Validation should explain absent sources and summarize large state gaps."""
    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (CompositeNetwork, None),
    )
    cfg = make_config("unused.ckpt")
    cfg["model"]["modules"]["encoder"].pop("weight_path")
    with pytest.raises(ValueError, match="at least one checkpoint"):
        export_model_weights(cfg, tmp_path / "empty.ckpt")

    class WideComposite(torch.nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.head = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(6)])

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (WideComposite, None),
    )
    encoder_path = tmp_path / "encoder.ckpt"
    save_component(encoder_path, torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match=r"\.\.\. \(2 more\)"):
        export_model_weights(
            make_config(encoder_path),
            tmp_path / "incomplete.ckpt",
        )
