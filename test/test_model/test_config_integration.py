"""End-to-end execution contracts for maintained model configurations."""

import math
from copy import deepcopy

import pandas as pd
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

    train_cfg = cfg.get("train")
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
def test_training_config_runs_checkpoint_validation(larcv_data, tmp_path, caplog):
    """Run training and on-the-fly validation through the real driver loop."""
    cfg = load_config_file(str(STANDALONE_MODEL_CONFIGS["uresnet"]), download=False)
    cfg = deepcopy(cfg)

    log_dir = tmp_path / "uresnet_validation"
    cfg["base"]["world_size"] = 0
    cfg["base"].pop("epochs", None)
    cfg["base"]["iterations"] = 1
    cfg["base"]["log_dir"] = str(log_dir)
    cfg["base"]["log_step"] = 1
    cfg["io"]["loader"]["minibatch_size"] = 1
    cfg["io"]["loader"]["num_workers"] = 0
    cfg["io"]["loader"]["dataset"]["file_keys"] = larcv_data
    cfg["io"]["loader"]["dataset"]["n_entry"] = 1
    cfg["model"]["weight_path"] = None
    cfg["train"]["weight_prefix"] = str(log_dir / "snapshot")
    cfg["train"].pop("save_epoch", None)
    cfg["train"]["save_step"] = 1
    cfg["validation"] = {"file_keys": larcv_data, "fraction": 0.01}

    with caplog.at_level("INFO", logger="spine"):
        Driver(cfg).run()

    train_log = log_dir / "train_log-0000000.csv"
    validation_log = log_dir / "validation_log-0000001.csv"
    assert train_log.is_file()
    assert validation_log.is_file()
    train_df = pd.read_csv(train_log)
    validation_df = pd.read_csv(validation_log)
    assert len(train_df) == len(validation_df) == 1
    assert not any(key.startswith("val_") for key in train_df.columns)
    assert {"loss", "accuracy"} <= set(validation_df.columns)

    output = "\n".join(caplog.messages)
    assert "VALIDATION START\nTraining iteration: 0" in output
    assert "Epoch:              1.000" in output
    assert "Batches:            1" in output
    assert "Val. 1/1" in output
    assert "Time (validation)" in output
    assert "VALIDATION COMPLETE\nTraining iteration: 0\nMetrics:" in output
    assert "  loss" in output
    assert "CHECKPOINT FINALIZATION\nTraining iteration: 0" in output
    separator = "=" * 69
    assert output.count(separator) == 6


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
