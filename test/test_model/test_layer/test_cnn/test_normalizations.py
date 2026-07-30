"""Tests for custom sparse activations and normalizations."""

import pytest
import torch

from spine.model.layer.cnn.nonlinearities import Mish
from spine.model.layer.cnn.normalizations import AdaIN, PixelNorm


class FeatureTensor:
    """Minimal sparse-tensor stand-in used by feature-only layers."""

    def __init__(self, features):
        self.features = features

    def replace_features(self, features):
        return FeatureTensor(features)


def test_pixel_norm_normalizes_each_feature_vector():
    tensor = FeatureTensor(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))

    output = PixelNorm()(tensor)

    assert torch.allclose(
        torch.linalg.vector_norm(output.features, dim=1), torch.ones(2)
    )
    assert repr(PixelNorm()) == "PixelNorm(eps=1e-08)"


def test_adain_handles_single_sparse_site_and_tracks_buffers():
    layer = AdaIN(2)
    tensor = FeatureTensor(torch.tensor([[2.0, 4.0]]))

    output = layer(tensor)

    assert torch.isfinite(output.features).all()
    assert {"_weight", "_bias"} <= dict(layer.named_buffers()).keys()


def test_adain_rejects_mismatched_controller_parameters():
    layer = AdaIN(2)

    with pytest.raises(ValueError, match="feature dimension"):
        layer.weight = torch.ones(3)


def test_mish_matches_dense_reference():
    features = torch.tensor([[-2.0, 0.5]])
    tensor = FeatureTensor(features)

    output = Mish()(tensor)

    expected = features * torch.tanh(torch.nn.functional.softplus(features))
    assert torch.allclose(output.features, expected)
