"""Tests for CLI model-module weight overrides."""

import pytest

from spine.bin.weight import apply_module_weight_overrides, parse_module_weights


def test_parse_module_weights():
    """Assignments should preserve module names and complete checkpoint paths."""
    assert parse_module_weights(
        ["uresnet_ppn=/weights/uresnet.ckpt", "graph_spice=run=4.ckpt"]
    ) == {
        "uresnet_ppn": "/weights/uresnet.ckpt",
        "graph_spice": "run=4.ckpt",
    }
    assert parse_module_weights(None) == {}


@pytest.mark.parametrize(
    "values",
    [
        ["uresnet_ppn"],
        ["=/weights/model.ckpt"],
        ["uresnet_ppn="],
        ["uresnet_ppn=a.ckpt", "uresnet_ppn=b.ckpt"],
    ],
)
def test_parse_module_weights_rejects_invalid_assignments(values):
    """Malformed and duplicate module assignments should fail clearly."""
    with pytest.raises(ValueError, match="--module-weight"):
        parse_module_weights(values)


def test_apply_module_weight_overrides():
    """Module paths should be applied independently of global model weights."""
    model_cfg = {
        "name": "full_chain",
        "weight_path": "full-chain.ckpt",
        "modules": {
            "uresnet_ppn": {"filters": 32},
            "graph_spice": {"shapes": [0, 1]},
        },
    }

    apply_module_weight_overrides(
        model_cfg,
        ["uresnet_ppn=uresnet.ckpt", "graph_spice=spice.ckpt"],
    )

    assert model_cfg["weight_path"] == "full-chain.ckpt"
    assert model_cfg["modules"]["uresnet_ppn"]["weight_path"] == "uresnet.ckpt"
    assert model_cfg["modules"]["graph_spice"]["weight_path"] == "spice.ckpt"


@pytest.mark.parametrize(
    ("model_cfg", "values", "match", "error"),
    [
        (
            {},
            ["graph_spice=weights.ckpt"],
            "requires a `model.modules` block",
            KeyError,
        ),
        (
            {"modules": "external.yaml"},
            ["graph_spice=weights.ckpt"],
            "must be an inline mapping",
            TypeError,
        ),
        (
            {"modules": {"uresnet_ppn": {}}},
            ["graph_spice=weights.ckpt"],
            "Unknown --module-weight module 'graph_spice'",
            KeyError,
        ),
        (
            {"modules": {"uresnet_ppn": "external.yaml"}},
            ["uresnet_ppn=weights.ckpt"],
            "model.modules.uresnet_ppn",
            TypeError,
        ),
    ],
)
def test_apply_module_weight_overrides_validates_model(model_cfg, values, match, error):
    """Overrides should require existing inline module configurations."""
    with pytest.raises(error, match=match):
        apply_module_weight_overrides(model_cfg, values)


def test_apply_module_weight_overrides_is_atomic():
    """A later invalid module should prevent all requested updates."""
    model_cfg = {"modules": {"uresnet_ppn": {}}}

    with pytest.raises(KeyError, match="missing"):
        apply_module_weight_overrides(
            model_cfg,
            ["uresnet_ppn=uresnet.ckpt", "missing=missing.ckpt"],
        )

    assert "weight_path" not in model_cfg["modules"]["uresnet_ppn"]


def test_apply_module_weight_overrides_ignores_empty_values():
    """Absent overrides should not require a modules block."""
    model_cfg = {}
    apply_module_weight_overrides(model_cfg, None)
    assert model_cfg == {}
