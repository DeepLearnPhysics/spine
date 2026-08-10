"""Construction and configuration contracts for supported models."""

from copy import deepcopy

import pytest

from spine.config import load_config_file
from spine.geo import GeoManager
from spine.model.factories import model_names, model_spec
from spine.model.manager import ModelManager
from spine.utils.conditional import TORCH_AVAILABLE

from .cases import EXAMPLE_CONFIGS, INFERENCE_MODEL_CONFIGS, MODEL_CONFIGS


@pytest.mark.parametrize(
    "config_path",
    EXAMPLE_CONFIGS,
    ids=lambda path: str(path.relative_to(path.parents[1])),
)
def test_example_config_uses_per_process_minibatch_size(config_path):
    """Keep example loader sizes independent of distributed world size."""
    cfg = load_config_file(str(config_path), download=False)
    loader_cfg = cfg["io"]["loader"]

    assert "minibatch_size" in loader_cfg
    assert "batch_size" not in loader_cfg


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


@pytest.mark.parametrize(
    "case_name",
    [
        "full_chain",
        "uresnet",
        "uresnet_bayes",
        "uresnet_ppn",
        "image_energy",
        "image_energy_ancestor",
        "image_pid",
        "image_pid_ancestor",
        "spice",
        "graph_spice",
        "grappa_inter",
        "grappa_shower",
        "grappa_track",
    ],
)
def test_prototype_training_schedule_is_epoch_based(case_name):
    """Prototype training duration and checkpoint cadence follow the dataset."""

    cfg = load_config_file(str(MODEL_CONFIGS[case_name]), download=False)
    base_cfg = cfg["base"]
    train_cfg = cfg["train"]

    assert "epochs" in base_cfg
    assert "iterations" not in base_cfg
    assert "save_epoch" in train_cfg
    assert "save_step" not in train_cfg


@pytest.mark.parametrize(
    "case_name",
    ["grappa_inter", "grappa_shower", "grappa_track"],
)
def test_grappa_train_and_test_config_contract(case_name):
    """Keep canonical GrapPA models and requested minibatch sizes aligned."""
    train_cfg = load_config_file(str(MODEL_CONFIGS[case_name]), download=False)
    test_cfg = load_config_file(
        str(INFERENCE_MODEL_CONFIGS[case_name]),
        download=False,
    )

    assert train_cfg["io"]["loader"]["minibatch_size"] == 64
    assert test_cfg["io"]["loader"]["minibatch_size"] == 2
    assert train_cfg["model"] == test_cfg["model"]


def test_full_chain_train_and_test_config_contract():
    """Keep canonical full-chain model definitions aligned and model-only."""
    train_cfg = load_config_file(str(MODEL_CONFIGS["full_chain"]), download=False)
    test_cfg = load_config_file(
        str(INFERENCE_MODEL_CONFIGS["full_chain"]),
        download=False,
    )

    assert set(train_cfg) == {"base", "io", "model", "train"}
    assert set(test_cfg) == {"base", "io", "model"}
    assert train_cfg["model"] == test_cfg["model"]
    assert train_cfg["io"]["loader"]["minibatch_size"] == 2
    assert test_cfg["io"]["loader"]["minibatch_size"] == 2


@pytest.mark.parametrize(
    "case_name",
    [
        "image_energy",
        "image_energy_ancestor",
        "image_pid",
        "image_pid_ancestor",
    ],
)
def test_image_train_and_test_config_contract(case_name):
    """Keep canonical image tasks aligned across training and inference."""
    train_cfg = load_config_file(str(MODEL_CONFIGS[case_name]), download=False)
    test_cfg = load_config_file(
        str(INFERENCE_MODEL_CONFIGS[case_name]),
        download=False,
    )

    assert train_cfg["io"]["loader"]["minibatch_size"] == 64
    assert test_cfg["io"]["loader"]["minibatch_size"] == 2
    assert train_cfg["model"] == test_cfg["model"]


@pytest.mark.parametrize(
    "case_name",
    ["image_energy_ancestor", "image_pid_ancestor"],
)
@pytest.mark.parametrize("config_group", [MODEL_CONFIGS, INFERENCE_MODEL_CONFIGS])
def test_ancestor_image_config_contract(case_name, config_group):
    """Keep tree construction and root-particle supervision explicit."""
    cfg = load_config_file(str(config_group[case_name]), download=False)
    model_cfg = cfg["model"]
    image_cfg = model_cfg["modules"]["image"]
    task_name = "energy" if case_name.startswith("image_energy") else "pid"
    loss_cfg = model_cfg["modules"]["image_loss"][task_name]

    assert model_cfg["network_input"]["object_data"] == "data"
    assert image_cfg["objects"]["source"] == "ancestor"
    assert loss_cfg["target_reduction"] == "ancestor"


@pytest.mark.parametrize("config_group", [MODEL_CONFIGS, INFERENCE_MODEL_CONFIGS])
def test_grappa_inter_parser_label_policy(config_group):
    """Keep interaction-identification labels on the requested truth policy."""
    cfg = load_config_file(str(config_group["grappa_inter"]), download=False)
    parser_cfg = cfg["io"]["loader"]["dataset"]["schema"]["data"]
    particle_cfg = parser_cfg["particle_info"]

    assert particle_cfg["type_include_secondary"] is False
    assert particle_cfg["type_include_mpr"] is False
    assert particle_cfg["primary_include_mpr"] is False


def test_grappa_track_has_no_unsupervised_node_head():
    """Do not construct trainable track outputs without a matching objective."""
    cfg = load_config_file(str(MODEL_CONFIGS["grappa_track"]), download=False)
    gnn_cfg = cfg["model"]["modules"]["grappa"]["gnn_model"]

    assert "node_pred" not in gnn_cfg


@pytest.mark.parametrize("config_group", [MODEL_CONFIGS, INFERENCE_MODEL_CONFIGS])
def test_graph_spice_keeps_cluster_truth_out_of_network_input(config_group):
    """Canonical Graph-SPICE networks must not consume instance truth."""
    cfg = load_config_file(str(config_group["graph_spice"]), download=False)
    model_cfg = cfg["model"]

    assert "clust_label" not in model_cfg["network_input"]
    assert model_cfg["loss_input"]["clust_label"] == "clust_label"
    assert not model_cfg["modules"]["graph_spice"]["constructor"].get(
        "label_edges",
        False,
    )


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
