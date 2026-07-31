"""Focused tests for point-proposal helpers and configuration."""

import pytest
import torch

from spine.model.uresnet_ppn.ppn import PPN, ExpandAs, PPNLoss
from spine.model.uresnet_ppn.vertex import VertexPPNLoss


class FeatureTensor:
    def __init__(self, features):
        self.features = features

    def replace_features(self, features):
        return FeatureTensor(features)


def test_expand_as_propagate_all_does_not_mutate_scores():
    features = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    tensor = FeatureTensor(features.clone())

    output = ExpandAs()(tensor, (2, 3), propagate_all=True)

    assert torch.equal(tensor.features, features)
    assert torch.equal(output.features, torch.ones((2, 3)))


def test_ppn_loss_normalizes_point_classes_to_tuple(cnn_config):
    loss = PPNLoss(cnn_config, {"point_classes": 2})

    assert loss.point_classes == (2,)


def test_ppn_takes_ghost_configuration_from_backbone(cnn_config):
    model = PPN({**cnn_config, "num_classes": 5, "ghost": True}, {})

    assert model.ghost
    assert model.masker is not None
    assert model.merge_concat is not None
    assert model.ghost_mask is not None


def test_ppn_rejects_true_ghost_mask_without_ghost_backbone(cnn_config):
    with pytest.raises(ValueError, match="UResNet `ghost: true`"):
        PPN(
            {**cnn_config, "num_classes": 5},
            {"use_true_ghost_mask": True},
        )


def test_ppn_rejects_duplicate_ghost_configuration(cnn_config):
    with pytest.raises(TypeError, match="ghost"):
        PPN(
            {**cnn_config, "num_classes": 5},
            {"ghost": True},
        )


def test_vertex_loss_reports_unsupported_label_contract():
    loss = VertexPPNLoss()

    with pytest.raises(NotImplementedError, match="vertex-label schema"):
        loss()
