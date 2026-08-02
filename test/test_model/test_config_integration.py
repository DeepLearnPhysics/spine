"""End-to-end execution contracts for maintained model configurations."""

import math
from copy import deepcopy

import pytest

from spine.config import load_config_file
from spine.driver import Driver
from spine.utils.conditional import LARCV_AVAILABLE, TORCH_AVAILABLE

from .cases import (
    EXPECTED_OUTPUTS,
    INFERENCE_MODEL_CONFIGS,
    STANDALONE_MODEL_CONFIGS,
)


def _as_float(value):
    """Convert a Python, NumPy, or scalar Torch value to float."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    elif hasattr(value, "item"):
        value = value.item()
    return float(value)


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(
    not (TORCH_AVAILABLE and LARCV_AVAILABLE),
    reason="The full model runtime and LArCV are required.",
)
@pytest.mark.parametrize("case_name", STANDALONE_MODEL_CONFIGS)
def test_model_config_runs_one_iteration(case_name, larcv_data, tmp_path):
    """Run one complete iteration and check the common training contract."""

    cfg = load_config_file(str(STANDALONE_MODEL_CONFIGS[case_name]), download=False)
    cfg = deepcopy(cfg)

    cfg["base"]["world_size"] = 0
    cfg["base"].pop("epochs", None)
    cfg["base"]["iterations"] = 1
    cfg["base"]["log_dir"] = str(tmp_path / case_name)
    cfg["io"]["loader"]["minibatch_size"] = 1
    cfg["io"]["loader"]["num_workers"] = 0
    cfg["io"]["loader"]["dataset"]["file_keys"] = larcv_data
    cfg["model"]["weight_path"] = None

    train_cfg = cfg["base"].get("train")
    if train_cfg is not None:
        train_cfg["weight_prefix"] = str(tmp_path / case_name / "snapshot")
        train_cfg["save_step"] = None

    result = Driver(cfg).process(iteration=0)

    assert "loss" in result
    assert EXPECTED_OUTPUTS[case_name] <= result.keys()
    assert math.isfinite(_as_float(result["loss"]))
    if case_name.startswith("image_energy"):
        assert math.isfinite(_as_float(result["energy_mae"]))
        assert math.isfinite(_as_float(result["energy_rmse"]))
        if case_name.endswith("_ancestor"):
            assert _as_float(result["energy_count"]) > 0
    else:
        assert math.isfinite(_as_float(result["accuracy"]))
        if case_name.endswith("_ancestor"):
            assert _as_float(result["pid_count"]) > 0


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(
    not (TORCH_AVAILABLE and LARCV_AVAILABLE),
    reason="The full model runtime and LArCV are required.",
)
@pytest.mark.parametrize(
    "case_name",
    [
        "graph_spice",
        "grappa_inter",
        "grappa_shower",
        "grappa_track",
        "image_energy",
        "image_energy_ancestor",
        "image_pid",
        "image_pid_ancestor",
        "spice",
        "uresnet_bayes",
        "uresnet_ppn",
    ],
)
def test_standalone_inference_config_runs_one_iteration(
    case_name,
    larcv_data,
    tmp_path,
):
    """Exercise a canonical standalone inference configuration."""
    cfg = load_config_file(
        str(INFERENCE_MODEL_CONFIGS[case_name]),
        download=False,
    )
    cfg = deepcopy(cfg)
    cfg["base"]["log_dir"] = str(tmp_path / f"{case_name}_inference")
    cfg["io"]["loader"]["minibatch_size"] = 1
    cfg["io"]["loader"]["num_workers"] = 0
    cfg["io"]["loader"]["dataset"]["file_keys"] = larcv_data
    cfg["model"]["weight_path"] = None

    result = Driver(cfg).process(iteration=0)

    assert EXPECTED_OUTPUTS[case_name] <= result.keys()
    assert math.isfinite(_as_float(result["loss"]))
    if case_name.startswith("image_energy"):
        assert math.isfinite(_as_float(result["energy_mae"]))
        assert math.isfinite(_as_float(result["energy_rmse"]))
        if case_name.endswith("_ancestor"):
            assert _as_float(result["energy_count"]) > 0
    else:
        assert math.isfinite(_as_float(result["accuracy"]))
        if case_name.endswith("_ancestor"):
            assert _as_float(result["pid_count"]) > 0
