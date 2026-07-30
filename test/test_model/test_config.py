"""Construction and configuration contracts for supported models."""

from copy import deepcopy

import pytest

from spine.config import load_config_file
from spine.geo import GeoManager
from spine.model.factories import model_names, model_spec
from spine.model.manager import ModelManager
from spine.utils.conditional import TORCH_AVAILABLE

from .cases import INFERENCE_MODEL_CONFIGS, MODEL_CONFIGS


@pytest.mark.parametrize("case_name", MODEL_CONFIGS)
def test_model_config_contract(case_name):
    """Every maintained configuration must describe a registered model."""

    cfg = load_config_file(str(MODEL_CONFIGS[case_name]), download=False)
    model_cfg = cfg["model"]

    assert model_cfg["name"] in model_names()
    assert isinstance(model_cfg["modules"], dict)
    assert isinstance(model_cfg["network_input"], dict)
    assert isinstance(model_cfg["loss_input"], dict)


@pytest.mark.parametrize("case_name", INFERENCE_MODEL_CONFIGS)
def test_inference_model_config_contract(case_name):
    """Every maintained inference configuration uses a registered model."""

    cfg = load_config_file(
        str(INFERENCE_MODEL_CONFIGS[case_name]),
        download=False,
    )
    model_cfg = cfg["model"]

    assert model_cfg["name"] in model_names()
    assert isinstance(model_cfg["modules"], dict)
    assert isinstance(model_cfg["network_input"], dict)
    assert isinstance(model_cfg["loss_input"], dict)


def test_supported_models_have_maintained_configs():
    """Every registered model must have at least one maintained configuration."""

    configured_models = {
        load_config_file(str(path), download=False)["model"]["name"]
        for path in MODEL_CONFIGS.values()
    }

    assert configured_models == set(model_names())


def test_uresnet_training_schedule_is_epoch_based():
    """Canonical training duration and checkpoint cadence follow the dataset."""

    cfg = load_config_file(str(MODEL_CONFIGS["uresnet"]), download=False)
    base_cfg = cfg["base"]
    train_cfg = base_cfg["train"]

    assert "epochs" in base_cfg
    assert "iterations" not in base_cfg
    assert "save_epoch" in train_cfg
    assert "save_step" not in train_cfg


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
@pytest.mark.parametrize("case_name", MODEL_CONFIGS)
def test_model_config_constructs_network_and_loss(case_name):
    """Build both halves of every supported model configuration."""

    cfg = load_config_file(str(MODEL_CONFIGS[case_name]), download=False)
    if "geo" in cfg:
        GeoManager.initialize_or_get(**cfg["geo"])

    model_cfg = cfg["model"]
    modules = model_cfg["modules"]
    original_modules = deepcopy(modules)
    spec = model_spec(model_cfg["name"])

    network_modules = ModelManager.select_network_modules(deepcopy(modules))
    network = spec.network(**network_modules)
    loss = spec.loss(**deepcopy(modules))

    assert modules == original_modules
    assert network.__class__ is spec.network
    assert loss.__class__ is spec.loss
