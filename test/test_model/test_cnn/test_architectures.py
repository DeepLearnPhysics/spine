"""Smoke tests for maintained sparse CNN architecture variants."""

import pytest

from spine.model import sparse
from spine.model.cnn.encoder import SparseResidualEncoder
from spine.model.cnn.factories import encoder_factory
from spine.model.cnn.fpn import FPN
from spine.model.cnn.mcdropout import (
    MCDropoutDecoder,
    MCDropoutEncoder,
)
from spine.model.cnn.mobilenet import MB3Encoder, MobileNetV3
from spine.model.cnn.senet import SENet
from spine.model.cnn.uresnet_layers import UResNet
from spine.model.cnn.uresnext import UResNeXt


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


def test_sparse_residual_encoder_uses_all_configured_features(
    cnn_config,
    sparse_table,
):
    """The pooled encoder must not silently discard extra input channels."""
    config = dict(cnn_config)
    config["num_input"] = 2
    table = sparse_table.new_empty((len(sparse_table), 6))
    table[:, :5] = sparse_table
    table[:, 5] = sparse_table[:, 4] * 2
    encoder = SparseResidualEncoder(feature_size=8, **config)

    output = encoder(table)

    assert output.shape == (2, 8)


def test_sparse_residual_encoder_sizes_coordinate_convolution(
    cnn_config,
    sparse_table,
):
    """Coordinate convolution should add its channels automatically."""
    encoder = SparseResidualEncoder(
        coord_conv=True,
        feature_size=8,
        **cnn_config,
    )

    output = encoder(sparse_table)

    assert output.shape == (2, 8)


def test_global_pooling_does_not_require_spatial_size(cnn_config, sparse_table):
    """Global pooling should work without an irrelevant detector extent."""
    config = dict(cnn_config)
    config.pop("spatial_size")
    encoder = SparseResidualEncoder(feature_size=8, **config)

    output = encoder(sparse_table)

    assert output.shape == (2, 8)


@pytest.mark.parametrize("pool_mode", ["sum", "max", "conv"])
def test_sparse_residual_encoder_pooling_modes(pool_mode, cnn_config, sparse_table):
    """Every advertised pooling backend returns one feature row per image."""
    encoder = SparseResidualEncoder(
        feature_size=8,
        pool_mode=pool_mode,
        **cnn_config,
    )

    assert encoder(sparse_table).shape == (2, 8)


def test_sparse_residual_encoder_validates_pooling_context(cnn_config):
    """Coordinate and convolutional pooling require a detector extent."""
    config = dict(cnn_config)
    config.pop("spatial_size")

    with pytest.raises(ValueError, match="spatial_size"):
        SparseResidualEncoder(coord_conv=True, **config)
    with pytest.raises(ValueError, match="spatial_size"):
        SparseResidualEncoder(pool_mode="conv", **config)
    with pytest.raises(ValueError, match="not recognized"):
        SparseResidualEncoder(pool_mode="median", **cnn_config)


def test_cnn_encoder_factory_constructs_registered_encoder(cnn_config):
    """The public encoder factory resolves the maintained CNN backend."""
    config = {"name": "cnn", "feature_size": 8, **cnn_config}
    assert isinstance(encoder_factory(config), SparseResidualEncoder)


@pytest.mark.parametrize("pool_mode", ["sum", "max", "none", "conv"])
def test_mc_dropout_encoder_pooling_variants(pool_mode, cnn_config, sparse_table):
    """Monte Carlo encoders support every documented pooling strategy."""
    encoder = MCDropoutEncoder(
        cnn_config,
        pool_mode=pool_mode,
        feature_size=8,
        add_classifier=pool_mode != "none",
    )
    output = encoder(sparse_table)

    if pool_mode == "none":
        assert output.shape[1] == encoder.num_planes[-1]
    else:
        assert output.shape == (2, 8)


def test_mc_dropout_components_validate_configuration(cnn_config):
    """Invalid stochastic-network dimensions fail during construction."""
    for dropout_p in (-0.1, 1.0):
        with pytest.raises(ValueError, match="dropout_p"):
            MCDropoutEncoder(cnn_config, dropout_p=dropout_p)
        with pytest.raises(ValueError, match="dropout_p"):
            MCDropoutDecoder(cnn_config, dropout_p=dropout_p)

    with pytest.raises(ValueError, match="feature_size"):
        MCDropoutEncoder(cnn_config, feature_size=0)
    with pytest.raises(ValueError, match="Unknown pooling"):
        MCDropoutEncoder(cnn_config, pool_mode="median")
    with pytest.raises(ValueError, match="encoder_filters"):
        MCDropoutDecoder(cnn_config, encoder_filters=0)


def test_mc_dropout_decoder_validates_feature_pyramid(cnn_config, sparse_table):
    """Decoder skip connections require the complete encoder pyramid."""
    encoder = MCDropoutEncoder(cnn_config)
    decoder = MCDropoutDecoder(cnn_config)
    sparse_input = sparse.SparseTensor(
        coordinates=sparse_table[:, :4].int(),
        features=sparse_table[:, 4:],
    )
    encoded = encoder.encode(sparse_input)

    with pytest.raises(ValueError, match="encoder tensors"):
        decoder(encoded["final_tensor"], encoded["encoder_tensors"][:-1])


@pytest.mark.parametrize(
    ("model_type", "kwargs"),
    [(UResNet, {}), (FPN, {}), (MobileNetV3, {}), (UResNeXt, {"cardinality": 2})],
)
def test_cnn_decoders_validate_skip_pyramid_length(model_type, kwargs, cnn_config):
    """Every decoder rejects incomplete encoder skip pyramids."""
    model = model_type(cnn_config, **kwargs)
    decode = model.decoder if hasattr(model, "decoder") else model.decode
    with pytest.raises(ValueError, match="encoder tensors"):
        decode(None, [])


def test_uresnext_and_senet_validate_architecture(cnn_config):
    """Grouped and squeeze-excitation networks reject invalid widths."""
    with pytest.raises(ValueError, match="cardinality"):
        UResNeXt(cnn_config, cardinality=0)
    with pytest.raises(ValueError, match="divisible"):
        UResNeXt(cnn_config, cardinality=3)
    with pytest.raises(ValueError, match=r"len\(dilations\)"):
        UResNeXt(cnn_config, cardinality=2, dilations=(1,))
    assert UResNeXt(cnn_config, cardinality=2).dilations == (1, 1)
    with pytest.raises(ValueError, match="se_ratio"):
        SENet(cnn_config, se_ratio=0)
