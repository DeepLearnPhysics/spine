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
