import os

import pytest

from spine.model.factories import model_factory


def test_model_construction(config_simple, xfail_models):
    """
    Tests whether a model and its loss can be constructed.
    """
    if config_simple["model"]["name"] in xfail_models:
        pytest.xfail(
            "%s is expected to fail at the moment." % config_simple["model"]["name"]
        )

    model, criterion = model_factory(config_simple["model"]["name"])
    net = model(config_simple["model"]["modules"])
    loss = criterion(config_simple["model"]["modules"])

    net.eval()
    net.train()
