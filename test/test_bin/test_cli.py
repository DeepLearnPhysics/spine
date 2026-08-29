"""Tests for the main SPINE command-line entry point."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from spine.bin import cli as cli_module


def test_main_updates_reader_config_and_runs(monkeypatch, tmp_path, capsys):
    """Command-line overrides should update reader configs before dispatch."""
    config_path = tmp_path / "train.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)

    def load_config(cfg_path):
        print("Resource:      /cache/model.ckpt (cached)")
        return {
            "base": {},
            "train": {},
            "io": {"reader": {"file_list": "stale.txt"}, "writer": {}},
            "model": {},
        }

    monkeypatch.setattr(cli_module, "load_config_file", load_config)
    monkeypatch.setattr(cli_module, "parse_value", lambda value: int(value))
    monkeypatch.setattr(
        cli_module,
        "set_nested_value",
        lambda cfg, key_path, value: (
            cfg | {"override": (key_path, value)},
            True,
        ),
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")

    cli_module.main(
        config=str(config_path),
        source=["a.root"],
        source_list=None,
        output="output.h5",
        output_dir="outputs",
        output_suffix="processed",
        n=12,
        nskip=3,
        entry_list="entries.txt",
        skip_entry_list="skip.txt",
        log_dir="logs",
        weight_prefix="weights",
        weight_path="weights.ckpt",
        weight_list="weights.txt",
        config_overrides=["io.batch_size=8"],
    )

    cfg = captured["cfg"]
    assert cfg["base"]["parent_path"] == str(tmp_path)
    assert cfg["base"]["log_dir"] == "logs"
    assert cfg["base"]["distributed"] is True
    assert cfg["base"]["world_size"] == 4
    assert cfg["train"]["weight_prefix"] == "weights"
    assert cfg["io"]["reader"]["file_keys"] == ["a.root"]
    assert cfg["io"]["reader"]["file_list"] is None
    assert cfg["io"]["reader"]["n_entry"] == 12
    assert cfg["io"]["reader"]["n_skip"] == 3
    assert cfg["io"]["reader"]["entry_list"] == "entries.txt"
    assert cfg["io"]["reader"]["skip_entry_list"] == "skip.txt"
    assert cfg["io"]["writer"]["file_name"] == "output.h5"
    assert cfg["io"]["writer"]["directory"] == "outputs"
    assert cfg["io"]["writer"]["suffix"] == "processed"
    assert cfg["model"]["weight_path"] == "weights.ckpt"
    assert cfg["model"]["weight_list"] == "weights.txt"
    assert cfg["override"] == ("io.batch_size", 8)
    output = capsys.readouterr().out
    assert "██████████" in output
    assert f"SPINE {cli_module.get_version()}" in output
    assert "DeepLearnPhysics Collaboration" in output
    assert "Startup\n-------" in output
    assert f"Configuration: {config_path}" in output
    assert (
        f"Configuration: {config_path}\n"
        "Resource:      /cache/model.ckpt (cached)\n\n"
    ) in output


def test_main_updates_loader_dataset(monkeypatch, tmp_path):
    """Loader-based configs should receive input overrides under dataset."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda cfg_path: {
            "io": {"loader": {"dataset": {"file_keys": ["stale.root"]}}},
            "model": {},
        },
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    cli_module.main(
        config=str(config_path),
        source=None,
        source_list="sources.txt",
        output=None,
        output_dir=None,
        output_suffix=None,
        n=None,
        nskip=None,
        entry_list=None,
        skip_entry_list=None,
        log_dir=None,
        weight_prefix=None,
        weight_path=None,
        weight_list=None,
        config_overrides=None,
    )

    assert captured["cfg"]["base"]["parent_path"] == str(tmp_path)
    assert captured["cfg"]["io"]["loader"]["dataset"]["file_keys"] is None
    assert captured["cfg"]["io"]["loader"]["dataset"]["file_list"] == "sources.txt"


@pytest.mark.parametrize(
    ("dataset", "source", "source_list", "expected"),
    [
        (
            {
                "name": "mixed",
                "larcv": {"file_keys": ["old.root"]},
                "hdf5": {"file_list": "old.txt"},
            },
            ["hdf5=/cache/a.h5", "hdf5=/cache/b.h5"],
            ["larcv=raw_files.txt"],
            {
                "larcv": {"file_keys": None, "file_list": "raw_files.txt"},
                "hdf5": {
                    "file_keys": ["/cache/a.h5", "/cache/b.h5"],
                    "file_list": None,
                },
            },
        ),
        (
            {
                "name": "joint",
                "base": {"file_keys": ["inherited.root"]},
                "primary": {"file_list": "old.txt"},
                "secondary": {"file_keys": ["old.root"]},
            },
            ["primary=/raw/main.root"],
            ["secondary=pileup.txt"],
            {
                "primary": {
                    "file_keys": ["/raw/main.root"],
                    "file_list": None,
                },
                "secondary": {
                    "file_keys": None,
                    "file_list": "pileup.txt",
                },
            },
        ),
    ],
)
def test_main_updates_composite_loader_sources(
    monkeypatch,
    tmp_path,
    dataset,
    source,
    source_list,
    expected,
):
    """Qualified inputs should update their mixed or joint source blocks."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {"io": {"loader": {"dataset": dataset}}, "model": {}},
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    cli_module.main(
        config=str(config_path),
        source=source,
        source_list=source_list,
        output=None,
        output_dir=None,
        output_suffix=None,
        n=None,
        nskip=None,
        entry_list=None,
        skip_entry_list=None,
        log_dir=None,
        weight_prefix=None,
        weight_path=None,
        weight_list=None,
        config_overrides=None,
    )

    result = captured["cfg"]["io"]["loader"]["dataset"]
    for target, target_expected in expected.items():
        assert result[target] == target_expected


@pytest.mark.parametrize(
    ("batch_size", "minibatch_size", "expected_key", "expected_value"),
    [
        (128, None, "batch_size", 128),
        (None, 32, "minibatch_size", 32),
    ],
)
@pytest.mark.parametrize(
    ("epochs", "iterations", "duration_key", "duration_value"),
    [
        (12.5, None, "epochs", 12.5),
        (None, 100, "iterations", 100),
    ],
)
def test_main_applies_runtime_resource_overrides(
    monkeypatch,
    tmp_path,
    batch_size,
    minibatch_size,
    expected_key,
    expected_value,
    epochs,
    iterations,
    duration_key,
    duration_value,
):
    """Resource flags should update their canonical configuration paths."""
    config_path = tmp_path / "train.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {
            "base": {
                "epochs": 2.0,
                "iterations": 20,
                "tensorboard": {"flush_secs": 5},
            },
            "io": {
                "loader": {
                    "dataset": {"file_keys": ["train.root"]},
                    "batch_size": 16,
                    "minibatch_size": 4,
                }
            },
            "model": {},
            "train": {},
        },
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    cli_module.main(
        config=str(config_path),
        source=None,
        source_list=None,
        output=None,
        output_dir=None,
        output_suffix=None,
        n=None,
        nskip=None,
        entry_list=None,
        skip_entry_list=None,
        log_dir="logs",
        weight_prefix=None,
        weight_path=None,
        weight_list=None,
        config_overrides=None,
        world_size=4,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        num_workers=8,
        epochs=epochs,
        iterations=iterations,
        tensorboard=True,
        tensorboard_dir="tb",
    )

    cfg = captured["cfg"]
    assert cfg["base"]["world_size"] == 4
    assert cfg["base"][duration_key] == duration_value
    alternate_duration = "iterations" if duration_key == "epochs" else "epochs"
    assert alternate_duration not in cfg["base"]
    assert cfg["base"]["log_dir"] == "logs"
    assert cfg["base"]["tensorboard"] == {"flush_secs": 5, "log_dir": "tb"}
    assert cfg["io"]["loader"][expected_key] == expected_value
    alternate_key = "minibatch_size" if expected_key == "batch_size" else "batch_size"
    assert alternate_key not in cfg["io"]["loader"]
    assert cfg["io"]["loader"]["num_workers"] == 8


def test_main_disables_tensorboard(monkeypatch, tmp_path):
    """The negative TensorBoard flag should replace existing writer options."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {
            "base": {"tensorboard": {"flush_secs": 5}},
            "io": {"reader": {}},
            "model": {},
        },
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    cli_module.main(
        config=str(config_path),
        source=None,
        source_list=None,
        output=None,
        output_dir=None,
        output_suffix=None,
        n=None,
        nskip=None,
        entry_list=None,
        skip_entry_list=None,
        log_dir=None,
        weight_prefix=None,
        weight_path=None,
        weight_list=None,
        config_overrides=None,
        tensorboard=False,
    )

    assert captured["cfg"]["base"]["tensorboard"] is False


def test_main_enables_tensorboard_with_custom_directory(monkeypatch, tmp_path):
    """Enabling TensorBoard with a directory should replace a false setting."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {
            "base": {"tensorboard": False},
            "io": {"reader": {}},
            "model": {},
        },
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    cli_module.main(
        config=str(config_path),
        source=None,
        source_list=None,
        output=None,
        output_dir=None,
        output_suffix=None,
        n=None,
        nskip=None,
        entry_list=None,
        skip_entry_list=None,
        log_dir=None,
        weight_prefix=None,
        weight_path=None,
        weight_list=None,
        config_overrides=None,
        tensorboard=True,
        tensorboard_dir="tb",
    )

    assert captured["cfg"]["base"]["tensorboard"] == {"log_dir": "tb"}


@pytest.mark.parametrize(
    ("val_source", "val_source_list", "expected_key", "expected_value"),
    [
        (["val_a.root", "val_b.root"], None, "file_keys", ["val_a.root", "val_b.root"]),
        (None, "validation.txt", "file_list", "validation.txt"),
    ],
)
def test_main_overrides_validation_source(
    monkeypatch,
    tmp_path,
    val_source,
    val_source_list,
    expected_key,
    expected_value,
):
    """Validation CLI inputs should replace the configured file selector."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {
            "io": {"loader": {"dataset": {"file_keys": ["train.root"]}}},
            "model": {},
            "train": {},
            "validation": {
                "file_keys": ["stale.root"],
                "file_list": "stale.txt",
                "fraction": 0.5,
            },
        },
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    cli_module.main(
        config=str(config_path),
        source=None,
        source_list=None,
        output=None,
        output_dir=None,
        output_suffix=None,
        n=None,
        nskip=None,
        entry_list=None,
        skip_entry_list=None,
        log_dir=None,
        weight_prefix=None,
        weight_path=None,
        weight_list=None,
        config_overrides=None,
        val_source=val_source,
        val_source_list=val_source_list,
    )

    validation = captured["cfg"]["validation"]
    assert validation[expected_key] == expected_value
    alternate_key = "file_list" if expected_key == "file_keys" else "file_keys"
    assert alternate_key not in validation
    assert validation["fraction"] == 0.5


def test_main_rejects_validation_source_for_inference(monkeypatch, tmp_path):
    """Validation-only source flags should not be accepted in inference mode."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {"io": {"reader": {}}, "model": {}},
    )

    with pytest.raises(ValueError, match="cannot be used with --inference"):
        cli_module.main(
            config=str(config_path),
            source=None,
            source_list=None,
            output=None,
            output_dir=None,
            output_suffix=None,
            n=None,
            nskip=None,
            entry_list=None,
            skip_entry_list=None,
            log_dir=None,
            weight_prefix=None,
            weight_path=None,
            weight_list=None,
            config_overrides=None,
            val_source=["validation.root"],
            inference=True,
        )


def test_main_validates_validation_source_overrides(monkeypatch, tmp_path):
    """Direct callers should receive clear validation-source errors."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {
            "io": {"loader": {"dataset": {"file_keys": ["train.root"]}}},
            "model": {},
            "validation": "invalid",
        },
    )

    common = {
        "config": str(config_path),
        "source": None,
        "source_list": None,
        "output": None,
        "output_dir": None,
        "output_suffix": None,
        "n": None,
        "nskip": None,
        "entry_list": None,
        "skip_entry_list": None,
        "log_dir": None,
        "weight_prefix": None,
        "weight_path": None,
        "weight_list": None,
        "config_overrides": None,
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli_module.main(
            **common,
            val_source=["validation.root"],
            val_source_list="validation.txt",
        )
    with pytest.raises(TypeError, match="`validation` block must be a mapping"):
        cli_module.main(**common, val_source=["validation.root"])


def test_main_validates_runtime_resource_overrides(monkeypatch, tmp_path):
    """Invalid combinations and launcher mismatches should fail clearly."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {"base": {}, "io": {"reader": {}}, "model": {}},
    )
    common = {
        "config": str(config_path),
        "source": None,
        "source_list": None,
        "output": None,
        "output_dir": None,
        "output_suffix": None,
        "n": None,
        "nskip": None,
        "entry_list": None,
        "skip_entry_list": None,
        "log_dir": None,
        "weight_prefix": None,
        "weight_path": None,
        "weight_list": None,
        "config_overrides": None,
    }

    with pytest.raises(ValueError, match="batch-size.*mutually exclusive"):
        cli_module.main(**common, batch_size=8, minibatch_size=4)
    with pytest.raises(ValueError, match="epochs.*mutually exclusive"):
        cli_module.main(**common, epochs=2.0, iterations=10)
    with pytest.raises(ValueError, match="tensorboard-dir.*no-tensorboard"):
        cli_module.main(**common, tensorboard=False, tensorboard_dir="tb")
    with pytest.raises(KeyError, match="require an `io.loader` block"):
        cli_module.main(**common, num_workers=4)

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(ValueError, match="conflicts with launcher WORLD_SIZE=8"):
        cli_module.main(**common, world_size=4)


def test_main_converts_to_inference_before_cli_overrides(monkeypatch, tmp_path):
    """The inference transform should run before authoritative CLI inputs."""
    config_path = tmp_path / "train.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {
            "base": {},
            "train": {},
            "io": {"reader": {"file_keys": ["training.root"]}},
            "model": {},
        },
    )

    def convert(config):
        assert config["io"]["reader"]["file_keys"] == ["training.root"]
        converted = dict(config)
        converted.pop("train")
        return converted

    monkeypatch.setattr(cli_module, "to_inference_config", convert)
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    cli_module.main(
        config=str(config_path),
        source=["inference.root"],
        source_list=None,
        output=None,
        output_dir=None,
        output_suffix=None,
        n=None,
        nskip=None,
        entry_list=None,
        skip_entry_list=None,
        log_dir=None,
        weight_prefix=None,
        weight_path=None,
        weight_list=None,
        config_overrides=None,
        inference=True,
    )

    assert "train" not in captured["cfg"]
    assert captured["cfg"]["io"]["reader"]["file_keys"] == ["inference.root"]


@pytest.mark.parametrize("resume", [True, False])
def test_main_applies_resume_override(monkeypatch, tmp_path, resume):
    """Dedicated resume flags should override the training configuration."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {
            "base": {},
            "io": {"reader": {}},
            "model": {},
            "train": {"resume": not resume},
        },
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    cli_module.main(
        config=str(config_path),
        source=None,
        source_list=None,
        output=None,
        output_dir=None,
        output_suffix=None,
        n=None,
        nskip=None,
        entry_list=None,
        skip_entry_list=None,
        log_dir=None,
        weight_prefix=None,
        weight_path=None,
        weight_list=None,
        config_overrides=None,
        resume=resume,
    )

    assert captured["cfg"]["train"]["resume"] is resume


def test_main_rejects_resume_without_training(monkeypatch, tmp_path):
    """Resume CLI flags require a training configuration."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda _path: {"io": {"reader": {}}, "model": {}},
    )

    with pytest.raises(KeyError, match="requires a `train` block"):
        cli_module.main(
            config=str(config_path),
            source=None,
            source_list=None,
            output=None,
            output_dir=None,
            output_suffix=None,
            n=None,
            nskip=None,
            entry_list=None,
            skip_entry_list=None,
            log_dir=None,
            weight_prefix=None,
            weight_path=None,
            weight_list=None,
            config_overrides=None,
            resume=True,
        )


def test_main_warns_when_output_options_have_no_writer(monkeypatch, tmp_path):
    """Output options should warn and be ignored when no writer is configured."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda cfg_path: {"io": {"reader": {}}, "model": {}},
    )
    monkeypatch.setattr("spine.main.run", lambda cfg: captured.setdefault("cfg", cfg))

    with pytest.warns(UserWarning, match="output options are ignored"):
        cli_module.main(
            config=str(config_path),
            source=None,
            source_list=None,
            output="output.h5",
            output_dir="outputs",
            output_suffix="processed",
            n=None,
            nskip=None,
            entry_list=None,
            skip_entry_list=None,
            log_dir=None,
            weight_prefix=None,
            weight_path=None,
            weight_list=None,
            config_overrides=None,
        )

    assert "writer" not in captured["cfg"]["io"]


def test_main_validation_errors(monkeypatch, tmp_path):
    """Main should reject malformed and incomplete runtime configuration."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)

    monkeypatch.setattr(cli_module, "load_config_file", lambda cfg_path: {"base": {}})
    with pytest.raises(KeyError, match="`io` block"):
        cli_module.main(
            config=str(config_path),
            source=None,
            source_list=None,
            output=None,
            output_dir=None,
            output_suffix=None,
            n=None,
            nskip=None,
            entry_list=None,
            skip_entry_list=None,
            log_dir=None,
            weight_prefix=None,
            weight_path=None,
            weight_list=None,
            config_overrides=None,
        )

    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda cfg_path: {"base": {}, "io": {}, "model": {}},
    )
    with pytest.raises(KeyError, match="`loader` or `reader`"):
        cli_module.main(
            config=str(config_path),
            source=["a.root"],
            source_list=None,
            output=None,
            output_dir=None,
            output_suffix=None,
            n=None,
            nskip=None,
            entry_list=None,
            skip_entry_list=None,
            log_dir=None,
            weight_prefix=None,
            weight_path=None,
            weight_list=None,
            config_overrides=None,
        )

    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda cfg_path: {"base": {}, "io": {"loader": {}}, "model": {}},
    )
    with pytest.raises(AssertionError, match="dataset"):
        cli_module.main(
            config=str(config_path),
            source=["a.root"],
            source_list=None,
            output=None,
            output_dir=None,
            output_suffix=None,
            n=None,
            nskip=None,
            entry_list=None,
            skip_entry_list=None,
            log_dir=None,
            weight_prefix=None,
            weight_path=None,
            weight_list=None,
            config_overrides=None,
        )

    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda cfg_path: {"base": {}, "io": {"reader": {}}, "model": {}},
    )
    with pytest.raises(KeyError, match="--weight_prefix"):
        cli_module.main(
            config=str(config_path),
            source=None,
            source_list=None,
            output=None,
            output_dir=None,
            output_suffix=None,
            n=None,
            nskip=None,
            entry_list=None,
            skip_entry_list=None,
            log_dir=None,
            weight_prefix="weights",
            weight_path=None,
            weight_list=None,
            config_overrides=None,
        )

    monkeypatch.setattr(cli_module, "parse_value", lambda value: value)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda cfg_path: {"base": {}, "io": {"reader": {}}, "model": {}},
    )
    with pytest.raises(ValueError, match="Invalid --set format"):
        cli_module.main(
            config=str(config_path),
            source=None,
            source_list=None,
            output=None,
            output_dir=None,
            output_suffix=None,
            n=None,
            nskip=None,
            entry_list=None,
            skip_entry_list=None,
            log_dir=None,
            weight_prefix=None,
            weight_path=None,
            weight_list=None,
            config_overrides=["bad_override"],
        )


def test_cli_entry_point_paths(monkeypatch):
    """The CLI should handle help, info, and main dispatch paths."""
    info_calls: list[str] = []
    main_calls: list[dict] = []

    monkeypatch.setattr(cli_module, "get_version", lambda: "9.9.9")
    monkeypatch.setattr(cli_module, "show_info", lambda: info_calls.append("info"))
    monkeypatch.setattr(cli_module, "main", lambda **kwargs: main_calls.append(kwargs))

    parser_help_called = {"value": False}

    class DummyParser:
        def print_help(self):
            parser_help_called["value"] = True

        def parse_args(self):
            return SimpleNamespace(
                info=False,
                config="config.yaml",
                source=["input.root"],
                source_list=None,
                val_source=["validation.root"],
                val_source_list=None,
                world_size=4,
                batch_size=None,
                minibatch_size=32,
                num_workers=8,
                epochs=None,
                iterations=100,
                tensorboard=True,
                tensorboard_dir="tb",
                output="out.h5",
                output_dir="outputs",
                output_suffix="processed",
                num_entries=2,
                nskip=1,
                entry_list="entries.txt",
                skip_entry_list="skip.txt",
                log_dir="logs",
                weight_prefix="weights",
                weight_path="weights.ckpt",
                weight_list="weights.txt",
                config_overrides=["a=1"],
                resume=None,
                inference=False,
            )

        def add_argument(self, *args, **kwargs):
            return None

        def add_mutually_exclusive_group(self):
            return self

    monkeypatch.setattr(
        argparse, "ArgumentParser", lambda *args, **kwargs: DummyParser()
    )

    monkeypatch.setattr(cli_module.sys, "argv", ["spine"])
    cli_module.cli()
    assert parser_help_called["value"] is True

    class InfoParser(DummyParser):
        def parse_args(self):
            args = super().parse_args()
            args.info = True
            return args

    monkeypatch.setattr(
        argparse, "ArgumentParser", lambda *args, **kwargs: InfoParser()
    )
    monkeypatch.setattr(cli_module.sys, "argv", ["spine", "--info"])
    cli_module.cli()
    assert info_calls == ["info"]

    class RunParser(DummyParser):
        pass

    monkeypatch.setattr(argparse, "ArgumentParser", lambda *args, **kwargs: RunParser())
    monkeypatch.setattr(cli_module.sys, "argv", ["spine", "-c", "config.yaml"])
    cli_module.cli()
    assert len(main_calls) == 1
    assert main_calls[0]["config"] == "config.yaml"
    assert main_calls[0]["source"] == ["input.root"]
    assert main_calls[0]["val_source"] == ["validation.root"]
    assert main_calls[0]["val_source_list"] is None
    assert main_calls[0]["world_size"] == 4
    assert main_calls[0]["minibatch_size"] == 32
    assert main_calls[0]["num_workers"] == 8
    assert main_calls[0]["epochs"] is None
    assert main_calls[0]["iterations"] == 100
    assert main_calls[0]["n"] == 2
    assert main_calls[0]["tensorboard"] is True
    assert main_calls[0]["tensorboard_dir"] == "tb"
    assert main_calls[0]["output_dir"] == "outputs"
    assert main_calls[0]["output_suffix"] == "processed"


def test_cli_parses_mixed_source_and_source_list(monkeypatch):
    """The two source flags may coexist when values carry target names."""
    calls = []
    monkeypatch.setattr(cli_module, "main", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        [
            "spine",
            "-c",
            "config.yaml",
            "--source-list",
            "larcv=raw_files.txt",
            "--source",
            "hdf5=/cache/*.h5",
        ],
    )

    cli_module.cli()

    assert len(calls) == 1
    assert calls[0]["source"] == ["hdf5=/cache/*.h5"]
    assert calls[0]["source_list"] == ["larcv=raw_files.txt"]
