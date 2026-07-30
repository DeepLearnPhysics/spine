"""Smoke tests for maintained sparse CNN architecture variants."""

import pytest

from spine.model import sparse
from spine.model.layer.cluster.factories import backbone_factory
from spine.model.layer.cnn.fpn import FPN
from spine.model.layer.cnn.mcdropout import (
    MCDropoutDecoder,
    MCDropoutEncoder,
)
from spine.model.layer.cnn.mobilenet import MB3Encoder, MobileNetV3
from spine.model.layer.cnn.senet import SENet
from spine.model.layer.cnn.uresnet_layers import UResNet
from spine.model.layer.cnn.uresnext import UResNeXt


@pytest.mark.parametrize("model_type", [UResNet, FPN, MobileNetV3, SENet])
def test_encoder_decoder_variants_run(model_type, cnn_config, sparse_table):
    model = model_type(cnn_config)

    output = model(sparse_table)

    assert isinstance(model, sparse.Network)
    assert len(output["encoder_tensors"]) == cnn_config["depth"] + 1
    assert len(output["decoder_tensors"]) == cnn_config["depth"] - 1
    assert output["decoder_tensors"][-1].features.shape[1] == cnn_config["filters"]


def test_uresnext_runs(cnn_config, sparse_table):
    model = UResNeXt(cnn_config, cardinality=2, dilations=(1, 2))

    output = model(sparse_table)

    assert isinstance(model, sparse.Network)
    assert output["decoder_tensors"][-1].features.shape[1] == cnn_config["filters"]


def test_mobile_encoder_does_not_call_nonexistent_decoder(cnn_config, sparse_table):
    model = MB3Encoder(cnn_config)

    output = model(sparse_table)

    assert isinstance(model, sparse.Network)
    assert set(output) == {"encoder_tensors", "final_tensor"}


def test_mc_dropout_encoder_decoder_share_contract(cnn_config, sparse_table):
    encoder = MCDropoutEncoder(
        cnn_config,
        dropout_p=0.2,
        feature_size=8,
    )
    decoder = MCDropoutDecoder(cnn_config, dropout_p=0.2)

    latent = encoder(sparse_table)

    assert isinstance(encoder, sparse.Network)
    assert isinstance(decoder, sparse.Network)
    assert latent.shape == (2, 8)

    # Exercise the segmentation path separately from pooled classification.
    input_tensor = sparse.SparseTensor(
        coordinates=sparse_table[:, :4].int(),
        features=sparse_table[:, 4:],
        batch_size=2,
    )
    encoded = encoder.encode(input_tensor)
    decoded = decoder(
        encoded["final_tensor"],
        encoded["encoder_tensors"],
    )
    assert decoded[-1].features.shape[1] == cnn_config["filters"]


def test_mc_dropout_rejects_invalid_layer_index(cnn_config):
    with pytest.raises(ValueError, match="outside"):
        MCDropoutEncoder(cnn_config, dropout_layers=[cnn_config["depth"]])


@pytest.mark.parametrize(
    ("name", "model_type"),
    [("uresnet", UResNet), ("fpn", FPN)],
)
def test_cluster_backbone_factory_uses_current_cnn_modules(
    name, model_type, cnn_config
):
    model = backbone_factory({"name": name, **cnn_config})

    assert isinstance(model, model_type)
