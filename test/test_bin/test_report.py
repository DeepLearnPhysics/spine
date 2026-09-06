"""Tests for the standalone spine-report command-line entry point."""

from __future__ import annotations

import pytest

from spine.bin.report import build_parser, cli
from spine.config import ConfigCycleError
from spine.vis.metric.report.manager import _load_config


def test_report_parser_requires_explicit_artifact_directories():
    """The reporter should expose the batch-friendly three-path contract."""
    args = build_parser().parse_args(
        [
            "--config",
            "report.yaml",
            "--input-dir",
            "raw",
            "--output-dir",
            "report",
        ]
    )

    assert args.config == "report.yaml"
    assert args.input_dir == "raw"
    assert args.output_dir == "report"


def test_report_cli_builds_report_and_prints_summary(monkeypatch, capsys):
    """The CLI should forward paths and identify the written summary."""
    calls = []

    def build_report(config, input_dir, output_dir):
        calls.append((config, input_dir, output_dir))
        return {"metrics": {"one": {}, "two": {}}}

    monkeypatch.setattr("spine.bin.report.build_report", build_report)

    status = cli(
        [
            "--config",
            "report.yaml",
            "--input-dir",
            "raw",
            "--output-dir",
            "report",
        ]
    )

    assert status == 0
    assert calls == [("report.yaml", "raw", "report")]
    assert "Wrote 2 metric summaries" in capsys.readouterr().out


def test_report_configuration_uses_standard_composition(tmp_path, monkeypatch):
    """Report loading should honor nested composition and config operations."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(config_dir))
    monkeypatch.setenv("REPORT_PATTERN", "metrics-*.csv")
    (config_dir / "base.yaml").write_text(
        """
__meta__:
  kind: fragment
strict: false
metrics:
  segmentation:
    name: segment_confusion
    source: ${REPORT_PATTERN}
""",
        encoding="utf-8",
    )
    (config_dir / "middle.yaml").write_text(
        """
__meta__:
  kind: fragment
include: base.yaml
metrics:
  obsolete:
    name: segment_confusion
    source: obsolete-*.csv
""",
        encoding="utf-8",
    )
    (config_dir / "report.yaml").write_text(
        """
include: middle.yaml
override:
  strict: true
  metrics-: obsolete
""",
        encoding="utf-8",
    )

    resolved, config = _load_config("report.yaml")

    assert resolved == config_dir / "report.yaml"
    assert config["strict"] is True
    assert config["metrics"]["segmentation"]["source"] == "metrics-*.csv"
    assert "obsolete" not in config["metrics"]


def test_report_configuration_rejects_circular_includes(tmp_path):
    """The report loader should retain standard include-cycle checks."""
    (tmp_path / "one.yaml").write_text("include: two.yaml\n", encoding="utf-8")
    (tmp_path / "two.yaml").write_text("include: one.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigCycleError, match="Circular include"):
        _load_config(tmp_path / "one.yaml")


def test_report_configuration_resolves_search_path_include(tmp_path, monkeypatch):
    """Report includes should fall back to shared configuration directories."""
    local_dir = tmp_path / "local"
    shared_dir = tmp_path / "shared"
    local_dir.mkdir()
    shared_dir.mkdir()
    monkeypatch.setenv("SPINE_CONFIG_PATH", str(shared_dir))
    (shared_dir / "shared.yaml").write_text(
        """
__meta__: {kind: fragment}
metrics:
  segmentation:
    name: segment_confusion
    source: metrics-*.csv
""",
        encoding="utf-8",
    )
    (local_dir / "report.yaml").write_text("include: shared.yaml\n", encoding="utf-8")

    _, config = _load_config(local_dir / "report.yaml")

    assert config["metrics"]["segmentation"]["name"] == "segment_confusion"
