"""Tests for the main SPINE command-line entry point."""

from __future__ import annotations

import argparse
import builtins
from types import ModuleType, SimpleNamespace

import pytest

from spine.bin import cli as cli_module


def test_main_updates_reader_config_and_runs(monkeypatch, tmp_path, capsys):
    """Command-line overrides should update reader configs before dispatch."""
    config_path = tmp_path / "train.yaml"
    config_path.write_text("io: {}\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli_module, "resolve_config_path", lambda cfg, current_dir: cfg)
    monkeypatch.setattr(
        cli_module,
        "load_config_file",
        lambda cfg_path: {
            "base": {"train": {}},
            "io": {"reader": {"file_list": "stale.txt"}, "writer": {}},
            "model": {},
        },
    )
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
    assert cfg["base"]["train"]["weight_prefix"] == "weights"
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
    assert "SPINE v" not in output


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
                output="out.h5",
                output_dir="outputs",
                output_suffix="processed",
                iterations=2,
                nskip=1,
                entry_list="entries.txt",
                skip_entry_list="skip.txt",
                log_dir="logs",
                weight_prefix="weights",
                weight_path="weights.ckpt",
                weight_list="weights.txt",
                config_overrides=["a=1"],
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
    assert main_calls[0]["output_dir"] == "outputs"
    assert main_calls[0]["output_suffix"] == "processed"


def test_get_version_show_info_and_dependency_checks(monkeypatch, capsys):
    """Version and info helpers should handle both available and missing deps."""
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "spine.version":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cli_module.get_version() == "unknown"

    monkeypatch.setattr(cli_module, "get_version", lambda: "1.2.3")
    original_check_dependencies = cli_module.check_dependencies
    monkeypatch.setattr(
        cli_module,
        "check_dependencies",
        lambda: {
            "torch": None,
            "matplotlib": "3.8.0",
            "plotly": None,
            "seaborn": "0.13.0",
            "minkowski": None,
        },
    )
    cli_module.show_info()
    output = capsys.readouterr().out
    assert "SPINE (Scalable Particle Imaging with Neural Embeddings) v1.2.3" in output
    assert "PyTorch not found" in output
    assert "Visualization: Not available" in output

    monkeypatch.setattr(cli_module, "check_dependencies", original_check_dependencies)

    def fake_missing_dep_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"torch", "matplotlib", "plotly", "seaborn", "MinkowskiEngine"}:
            raise ImportError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_missing_dep_import)
    monkeypatch.setattr(
        cli_module,
        "package_version",
        lambda name: (_ for _ in ()).throw(cli_module.PackageNotFoundError(name)),
    )
    deps = cli_module.check_dependencies()
    assert deps["torch"] is None
    assert deps["minkowski"] is None
    assert set(deps) == {"torch", "matplotlib", "plotly", "seaborn", "minkowski"}


def test_get_version_and_dependency_checks_success(monkeypatch):
    """Version lookup and dependency probes should report installed modules."""
    from spine.version import __version__

    original_import = builtins.__import__

    def fake_dep_import(name, globals=None, locals=None, fromlist=(), level=0):
        versions = {
            "torch": "2.0.0",
            "matplotlib": "3.8.0",
            "plotly": "5.0.0",
            "seaborn": "0.13.0",
            "MinkowskiEngine": "0.5.4",
        }
        if name in versions:
            module = ModuleType(name)
            module.__version__ = versions[name]
            return module
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_dep_import)
    monkeypatch.setattr(
        cli_module,
        "package_version",
        lambda name: {"MinkowskiEngine": "0.5.4"}[name],
    )

    assert cli_module.get_version() == __version__
    deps = cli_module.check_dependencies()
    assert deps["torch"] == "2.0.0"
    assert deps["matplotlib"] == "3.8.0"
    assert deps["plotly"] == "5.0.0"
    assert deps["seaborn"] == "0.13.0"
    assert deps["minkowski"] == "0.5.4"


def test_show_info_reports_available_optional_features(monkeypatch, capsys):
    """Info output should report available model and visualization extras."""
    monkeypatch.setattr(cli_module, "get_version", lambda: "1.2.3")
    monkeypatch.setattr(
        cli_module,
        "check_dependencies",
        lambda: {
            "torch": "2.0.0",
            "matplotlib": "3.8.0",
            "plotly": "5.0.0",
            "seaborn": None,
            "minkowski": None,
        },
    )

    cli_module.show_info()

    output = capsys.readouterr().out
    assert "Model: Neural networks available (PyTorch 2.0.0)" in output
    assert "Visualization: Available (Plotly 5.0.0)" in output
