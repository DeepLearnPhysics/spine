"""Integration of generic named-list edits with full-chain stage planning."""

from copy import deepcopy

import yaml

from spine.config import load_config_file
from spine.model.full_chain.config import build_chain_plan


def test_modifier_inserts_calibrations_without_changing_existing_stages(tmp_path):
    """Loaded edits produce the intended plan while preserving stage options."""
    deghost = {
        "name": "deghosting",
        "provider": "deghost",
        "config": {"mode": "label", "charge_rescaling": "collection"},
    }
    segmentation = {
        "name": "segmentation",
        "provider": "segmentation",
        "config": {"mode": "label"},
    }
    modules = {"chain": {"stages": [deghost, segmentation]}}
    original = deepcopy(modules)
    base = {"__meta__": {"kind": "fragment"}, "model": {"modules": modules}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    predeghost = {
        "name": "predeghost_scale",
        "provider": "calibration",
        "config": {"mode": "apply", "calibration": {"gain": {"gain": 1.1}}},
    }
    presegmentation = {
        "name": "presegmentation_calibration",
        "provider": "calibration",
        "config": {"mode": "apply", "calibration": {"gain": {"gain": 2.0}}},
    }
    modifier = {
        "__meta__": {"kind": "fragment"},
        "override": {
            "model.modules.chain.stages~": [
                {"insert": {"before": "deghosting", "value": predeghost}},
                {"insert": {"before": "segmentation", "value": presegmentation}},
            ]
        },
    }
    (tmp_path / "modifier.yaml").write_text(yaml.safe_dump(modifier))
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text("include: [base.yaml, modifier.yaml]\n")

    loaded = load_config_file(str(bundle))["model"]["modules"]
    assert loaded["chain"]["stages"] == [
        predeghost,
        deghost,
        presegmentation,
        segmentation,
    ]
    plan = build_chain_plan(loaded["chain"], loaded)
    assert [stage.name for stage in plan] == [
        "predeghost_scale",
        "deghosting",
        "presegmentation_calibration",
        "segmentation",
    ]
    assert plan[0].config["calibration"]["gain"]["gain"] == 1.1
    assert plan[1].config["charge_rescaling"] == "collection"
    assert plan[2].config["calibration"]["gain"]["gain"] == 2.0
    assert load_config_file(str(tmp_path / "base.yaml"))["model"]["modules"] == original


def test_full_chain_provider_defaults_to_instance_name():
    """Shorthand stages remain addressable by name before plan construction."""
    from spine.config import load_config

    cfg = load_config("""
chain:
  stages:
    - name: deghost
      config: {mode: label}
override:
  chain.stages~:
    insert:
      before: deghost
      value:
        name: calibration
        config: {mode: label}
""")
    plan = build_chain_plan(cfg["chain"], {})
    assert [(stage.name, stage.provider) for stage in plan] == [
        ("calibration", "calibration"),
        ("deghost", "deghost"),
    ]
    assert all(stage.config == {"mode": "label"} for stage in plan)


def test_full_chain_provider_shorthand_stays_strict():
    """Only an absent provider defaults; names are always required and unique."""
    import pytest

    for provider in (None, "", " ", 1):
        with pytest.raises(ValueError, match="provider names"):
            build_chain_plan(
                {"stages": [{"name": "deghost", "provider": provider}]}, {}
            )
    with pytest.raises(ValueError, match="requires `name`"):
        build_chain_plan({"stages": [{"provider": "deghost"}]}, {})
    with pytest.raises(ValueError, match="Duplicate"):
        build_chain_plan(
            {
                "stages": [
                    {"name": "deghost"},
                    {"name": "deghost", "provider": "calibration"},
                ]
            },
            {},
        )


def test_full_chain_keeps_legacy_mixed_parameter_precedence():
    """Sharing extraction preserves FullChain's inline-over-nested behavior."""
    from copy import deepcopy

    stage = {
        "name": "deghost",
        "uses": ["network"],
        "loss": "loss_block",
        "config": {"mode": "label", "network": {"nested": True}, "values": [1]},
        "mode": "uresnet",
        "network": {"inline": True},
    }
    original = deepcopy(stage)
    plan = build_chain_plan(
        {"stages": [stage]},
        {"network": {"base": True}, "loss_block": {"weight": 1}},
        require_losses=True,
    )
    assert plan[0].config == {
        "mode": "uresnet",
        "network": {"inline": True},
        "values": [1],
    }
    assert plan[0].loss_config == {"weight": 1}
    plan[0].config["values"].append(2)
    assert stage == original
