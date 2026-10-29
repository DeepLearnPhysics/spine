"""End-to-end execution contracts for maintained model configurations."""

import math
from copy import deepcopy

import pandas as pd
import pytest

from spine.config import load_config_file
from spine.driver import Driver
from spine.utils.conditional import LARCV_AVAILABLE, TORCH_AVAILABLE, torch

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
@pytest.mark.parametrize(
    "case_name", ["uresnet", "uresnet_ppn", "graph_spice", "spice"]
)
def test_sparse_model_config_runs_with_lattice_phase(case_name, larcv_data, tmp_path):
    """Exercise lattice phase through each maintained UResNet-backed task."""
    cfg = load_config_file(str(STANDALONE_MODEL_CONFIGS[case_name]), download=False)
    cfg = deepcopy(cfg)

    cfg["base"]["world_size"] = 0
    cfg["base"].pop("epochs", None)
    cfg["base"]["iterations"] = 1
    cfg["base"]["log_dir"] = str(tmp_path / f"{case_name}_lattice")
    cfg["io"]["loader"]["minibatch_size"] = 1
    cfg["io"]["loader"]["num_workers"] = 0
    cfg["io"]["loader"]["dataset"]["file_keys"] = larcv_data
    cfg["model"]["weight_path"] = None

    modules = cfg["model"]["modules"]
    if case_name in {"uresnet", "uresnet_ppn"}:
        uresnet = modules["uresnet"]
    elif case_name == "graph_spice":
        uresnet = modules["graph_spice"]["embedder"]["uresnet"]
    else:
        uresnet = modules[case_name]["uresnet"]
    uresnet["lattice"] = {"period": "auto"}

    cfg["train"]["weight_prefix"] = str(tmp_path / f"{case_name}_lattice" / "snapshot")
    cfg["train"]["save_step"] = None

    result = Driver(cfg).process(iteration=0)

    assert math.isfinite(_as_float(result["loss"]))


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(
    not (TORCH_AVAILABLE and LARCV_AVAILABLE),
    reason="The full model runtime and LArCV are required.",
)
@pytest.mark.parametrize("initial", [True, False])
def test_training_config_runs_checkpoint_validation(
    larcv_data, tmp_path, caplog, initial
):
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
    cfg["validation"] = {
        "file_keys": larcv_data,
        "fraction": 0.01,
        "early_stopping": {"monitor": "loss", "patience": 2},
        "best_checkpoint": True,
    }
    # Exercise the default initial baseline as well as the explicit opt-out.
    if not initial:
        cfg["validation"]["initial"] = False

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
    initial_log = log_dir / "validation_log-0000000.csv"
    initial_checkpoint = log_dir / "snapshot-initial.ckpt"
    if initial:
        assert initial_log.is_file()
        assert initial_checkpoint.is_file()
        assert (log_dir / "snapshot-initial.ckpt.sha256").is_file()
        initial_df = pd.read_csv(initial_log)
        assert len(initial_df) == 1
        assert initial_df["epoch"].iloc[0] == 0.0
        baseline = torch.load(initial_checkpoint, weights_only=True)
        assert baseline["global_step"] == -1
        assert baseline["global_epoch"] == 0.0
        assert baseline["validation"]["early_stopping"]["bad_checks"] == 0
        baseline_loss = baseline["validation"]["metrics"]["loss"]
        assert baseline_loss == pytest.approx(initial_df["loss"].iloc[0])
    else:
        assert not initial_log.exists()
        assert not initial_checkpoint.exists()
        baseline_loss = math.inf

    best = torch.load(log_dir / "snapshot-best.ckpt", weights_only=True)
    assert best["validation"]["metrics"]["loss"] == pytest.approx(
        min(baseline_loss, validation_df["loss"].iloc[0])
    )

    output = "\n".join(caplog.messages)
    assert "CHECKPOINT\nReason:             scheduled\nTraining iteration: 0" in output
    assert "Epoch:              1.000" in output
    assert "Validation batches: 1" in output
    assert "VALIDATION START" in output
    assert "Val. 1/1" in output
    assert "Time (validation)" in output
    assert "VALIDATION COMPLETE\nMetrics:" in output
    assert "  loss" in output
    assert "EARLY STOPPING" in output
    assert "Monitor         : loss (min)" in output
    assert "Status          : initial best" in output
    assert "Patience        : 0/2 (reset)" in output
    assert "Saving checkpoint..." in output
    assert f"Checkpoint saved: {log_dir / 'snapshot-0.ckpt'}" in output
    separator = "=" * 69
    assert output.count(separator) == 2 * (1 + int(initial))
    train_summary = output.index("Iter. 0 (epoch 1.000)")
    checkpoint = output.index("CHECKPOINT\nReason:             scheduled")
    assert train_summary < checkpoint
    initial_header = "CHECKPOINT\nReason:             initial\nTraining iteration: -1"
    if initial:
        assert output.count(initial_header) == 1
        initial_saved = output.index(f"Checkpoint saved: {initial_checkpoint}")
        assert output.index(initial_header) < initial_saved < train_summary
    else:
        assert initial_header not in output


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
