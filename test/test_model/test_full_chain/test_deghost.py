"""Tests for the full-chain deghosting provider."""

import numpy as np
import pytest
import torch

from spine.constants import GHOST_SHP
from spine.data import ClusterLabelBatch, TensorBatch
from spine.model.full_chain.providers import deghost as deghost_module
from spine.model.full_chain.providers.deghost import (
    DeghostLossStage,
    DeghostStage,
    build_deghost_loss,
    build_deghost_stage,
)
from spine.model.full_chain.state import ChainState


def make_data() -> TensorBatch:
    """Build four canonical voxels in one event."""
    rows = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 2],
            [0, 2, 0, 0, 3],
            [0, 3, 0, 0, 4],
        ],
        dtype=torch.float32,
    )
    return TensorBatch(
        rows,
        counts=[4],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )


def make_cluster_label() -> ClusterLabelBatch:
    """Build compact truth values aligned with :func:`make_data`."""
    rows = torch.tensor(
        [
            [0, 0, 0, 0, 10, 0],
            [0, 1, 0, 0, 20, 0],
            [0, 2, 0, 0, 30, 1],
            [0, 3, 0, 0, 40, 1],
        ],
        dtype=torch.float32,
    )
    data = TensorBatch(
        rows,
        counts=[4],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    return ClusterLabelBatch(data)


def make_seg_label() -> TensorBatch:
    """Build alternating physical and ghost semantic labels."""
    return TensorBatch(
        torch.tensor([0, GHOST_SHP, 1, GHOST_SHP]),
        counts=[4],
    )


def test_label_deghosting_aligns_truth_sources_and_charge() -> None:
    """One truth mask drives data, source, index, and charge adaptation."""
    sources = TensorBatch(torch.arange(8).reshape(4, 2), counts=[4])
    stage = DeghostStage("deghost", "label", None, "label")

    result = stage(
        ChainState(
            data=make_data(),
            sources=sources,
            seg_label=make_seg_label(),
            clust_label=make_cluster_label(),
        )
    )

    adapted = result.products["point_data"]
    assert adapted.data.counts.tolist() == [2]
    assert adapted.data.values.torch_tensor().tolist() == [10.0, 30.0]
    assert adapted.data_q is adapted.data
    assert adapted.orig_index.index.tolist() == [0, 2]
    assert adapted.sources.torch_tensor().tolist() == [[0, 1], [4, 5]]
    assert result.outputs["orig_index_label"].index.tolist() == [0, 2]
    assert result.outputs["sources_label"].torch_tensor().tolist() == [[0, 1], [4, 5]]


def test_learned_deghosting_publishes_scores_and_rescales_charge() -> None:
    """Learned masks publish logits and may apply reconstructed charge scaling."""

    class Model:
        def __call__(self, _data):
            return {
                "segmentation": TensorBatch(
                    torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
                    counts=[4],
                )
            }

    stage = DeghostStage("deghost", "uresnet", Model(), "average")
    stage.charge_rescaler = lambda _data: torch.tensor([100.0, 200.0])
    result = stage(ChainState(data=make_data()))

    assert result.outputs["ghost"] is not None
    assert result.outputs["ghost_pred"].torch_tensor().tolist() == [0, 1, 0, 1]
    adapted = result.products["point_data"]
    assert adapted.data.values.torch_tensor().tolist() == [100.0, 200.0]
    assert adapted.data_q is adapted.data


def test_deghost_stage_validates_modes_and_inputs() -> None:
    """Invalid mode combinations and unavailable truth fail immediately."""
    with pytest.raises(ValueError, match="mode"):
        DeghostStage("deghost", None, None, None)
    with pytest.raises(ValueError, match="charge-rescaling"):
        DeghostStage("deghost", "label", None, "bad")
    with pytest.raises(ValueError, match="requires label deghosting"):
        DeghostStage("deghost", "uresnet", None, "label")
    with pytest.raises(ValueError, match="requires `seg_label`"):
        DeghostStage("deghost", "label", None, None)(ChainState(data=make_data()))
    with pytest.raises(ValueError, match="requires `clust_label`"):
        DeghostStage("deghost", "label", None, "label")(
            ChainState(data=make_data(), seg_label=make_seg_label())
        )
    with pytest.raises(RuntimeError, match="not initialized"):
        DeghostStage("deghost", "uresnet", None, None)(ChainState(data=make_data()))


def test_deghost_loss_adapts_binary_labels() -> None:
    """The loss adapter converts semantic truth into binary ghost targets."""

    class Loss:
        def __init__(self):
            self.labels = None

        def __call__(self, *, seg_label, segmentation):
            self.labels = seg_label
            return {"loss": torch.tensor(0.0), "accuracy": 1.0}

    loss = Loss()
    stage = DeghostLossStage("deghost", loss)
    scores = TensorBatch(torch.zeros((4, 2)), counts=[4])
    result = stage({"seg_label": make_seg_label(), "ghost": scores})

    assert loss.labels.torch_tensor().tolist() == [0, 1, 0, 1]
    assert result["accuracy"] == 1.0
    with pytest.raises(ValueError, match="requires `seg_label` and `ghost`"):
        stage({"seg_label": make_seg_label()})


def test_deghost_builders_validate_and_register_native_modules(monkeypatch) -> None:
    """Provider builders own configured model and objective modules."""

    class FakeModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    class FakeLoss(torch.nn.Module):
        def __init__(self, model, loss):
            super().__init__()
            self.config = (model, loss)

    monkeypatch.setattr(deghost_module, "UResNetSegmentation", FakeModel)
    monkeypatch.setattr(deghost_module, "SegmentationLoss", FakeLoss)

    owner = torch.nn.Module()
    with pytest.raises(ValueError, match="requires `uresnet_deghost`"):
        build_deghost_stage("deghost", {"mode": "uresnet"}, owner)
    stage = build_deghost_stage(
        "deghost",
        {"mode": "uresnet", "uresnet_deghost": {"depth": 2}},
        owner,
    )
    assert owner.uresnet_deghost is stage.model

    assert build_deghost_loss("deghost", {}, owner) is None
    with pytest.raises(TypeError, match="must be mappings"):
        build_deghost_loss("deghost", {"uresnet_deghost": [], "loss": {}}, owner)
    loss_stage = build_deghost_loss(
        "deghost",
        {"uresnet_deghost": {"depth": 2}, "loss": {"balance_loss": False}},
        owner,
    )
    assert owner.deghost_loss is loss_stage.loss
