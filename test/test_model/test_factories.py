"""Tests for top-level model discovery and registry contracts."""

from types import SimpleNamespace

import pytest

from spine.model.factories import model_dict, model_factory, model_names, model_spec
from spine.model.registry import ModelSpec


def test_model_discovery_is_dependency_light():
    """Supported names can be listed without importing implementations."""
    assert model_names() == (
        "full_chain",
        "graph_spice",
        "grappa",
        "image",
        "spice",
        "uresnet",
        "uresnet_bayes",
        "uresnet_ppn",
    )


def test_unknown_model_reports_available_names():
    """Registry errors make the supported choices discoverable."""
    try:
        model_spec("not_a_model")
    except ValueError as err:
        message = str(err)
    else:
        raise AssertionError("Unknown model name was accepted.")

    assert "not_a_model" in message
    assert "uresnet" in message


def test_model_factory_and_compatibility_dictionary_resolve_specs():
    """Public factory forms return the network/loss pairs in registry order."""
    network, loss = model_factory("uresnet")
    models = model_dict()

    assert models["uresnet"] == (network, loss)
    assert tuple(models) == model_names()


@pytest.mark.parametrize(
    ("module", "error", "message"),
    [
        (SimpleNamespace(__name__="bad"), RuntimeError, "does not define"),
        (
            SimpleNamespace(__name__="bad", MODEL_SPEC="bad"),
            TypeError,
            "must be a ModelSpec",
        ),
        (
            SimpleNamespace(
                __name__="bad",
                MODEL_SPEC=ModelSpec("wrong", object),
            ),
            ValueError,
            "does not match",
        ),
    ],
)
def test_model_spec_validates_imported_module_contract(
    monkeypatch,
    module,
    error,
    message,
):
    """Malformed lazily imported model modules fail at the registry boundary."""
    monkeypatch.setattr("spine.model.factories.import_module", lambda _name: module)
    with pytest.raises(error, match=message):
        model_spec("uresnet")
