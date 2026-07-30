"""Tests for common CNN configuration parsing."""

from types import SimpleNamespace

import pytest

from spine.model.layer.cnn.configuration import setup_cnn_configuration


def test_setup_cnn_configuration_stores_canonical_names(cnn_config):
    target = SimpleNamespace()

    setup_cnn_configuration(target, **cnn_config)

    assert target.dimension == 3
    assert target.num_filters == 4
    assert target.num_planes == [4, 8]
    assert target.act_cfg == "relu"
    assert target.norm_cfg == "none"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("reps", 0),
        ("depth", 0),
        ("filters", 0),
        ("input_kernel", 0),
        ("data_dim", 0),
        ("num_input", 0),
        ("spatial_size", 0),
    ],
)
def test_setup_cnn_configuration_rejects_invalid_sizes(cnn_config, name, value):
    cnn_config[name] = value

    with pytest.raises(ValueError, match=name):
        setup_cnn_configuration(SimpleNamespace(), **cnn_config)
