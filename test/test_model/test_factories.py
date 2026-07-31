"""Tests for top-level model discovery and registry contracts."""

from spine.model.factories import model_names, model_spec


def test_model_discovery_is_dependency_light():
    """Supported names can be listed without importing implementations."""
    assert model_names() == (
        "full_chain",
        "graph_spice",
        "grappa",
        "image_class",
        "spice",
        "uresnet",
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
