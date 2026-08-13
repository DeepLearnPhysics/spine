"""Tests for deterministic training-to-inference configuration transforms."""

from copy import deepcopy

import pytest

from spine.config.inference import get_inference_cfg, to_inference_config


def training_config():
    """Return a representative loader-backed training configuration."""
    return {
        "base": {"epochs": 12, "world_size": 2},
        "io": {
            "loader": {
                "batch_size": 8,
                "shuffle": True,
                "sampler": {"name": "random_sequence"},
                "num_workers": 4,
                "dataset": {"file_keys": ["train.root"]},
            },
            "writer": {"name": "hdf5"},
        },
        "model": {"name": "dummy"},
        "train": {"weight_prefix": "snapshot"},
        "validation": {"iterations": 5},
    }


def test_to_inference_config_is_independent_and_applies_overrides():
    """Conversion must be deterministic, non-mutating, and overrideable."""
    config = training_config()
    original = deepcopy(config)

    result = to_inference_config(
        config,
        file_keys=["inference.root"],
        weight_path="snapshot.ckpt",
        batch_size=2,
        num_workers=0,
        cpu=True,
    )

    assert config == original
    assert "train" not in result and "validation" not in result
    assert "epochs" not in result["base"]
    assert result["base"]["iterations"] == -1
    assert result["base"]["world_size"] == 0
    assert result["base"]["unwrap"] is True
    assert result["model"]["weight_path"] == "snapshot.ckpt"
    loader = result["io"]["loader"]
    assert loader["shuffle"] is False and "sampler" not in loader
    assert loader["batch_size"] == 2 and loader["num_workers"] == 0
    assert loader["dataset"]["file_keys"] == ["inference.root"]


def test_to_inference_config_preserves_explicit_iterations_without_consumers():
    """Existing iteration limits remain authoritative for raw model inference."""
    config = training_config()
    config["base"]["iterations"] = 7
    config["io"].pop("writer")

    result = to_inference_config(config)

    assert result["base"]["iterations"] == 7
    assert "unwrap" not in result["base"]


def test_to_inference_config_validates_contextual_overrides():
    """Loader and model overrides require the corresponding configuration."""
    with pytest.raises(ValueError, match="Loader overrides"):
        to_inference_config({"base": {}, "io": {}}, batch_size=2)
    with pytest.raises(ValueError, match="weight override"):
        to_inference_config(
            {"base": {}, "io": {"loader": {"dataset": {}}}},
            weight_path="snapshot.ckpt",
        )


def test_legacy_inference_wrapper_loads_yaml(tmp_path):
    """The deprecated file-loading wrapper remains usable during migration."""
    path = tmp_path / "training.yaml"
    path.write_text("base: {}\nio: {}\n", encoding="utf-8")

    with pytest.deprecated_call(match="to_inference_config"):
        result = get_inference_cfg(path)

    assert result["base"]["iterations"] == -1


def test_legacy_inference_wrapper_accepts_a_mapping():
    """The compatibility wrapper should retain its original mapping input API."""
    with pytest.deprecated_call(match="to_inference_config"):
        result = get_inference_cfg({"base": {}, "io": {}})

    assert result["base"]["iterations"] == -1
