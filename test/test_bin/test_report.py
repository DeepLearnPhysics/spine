"""Tests for the standalone spine-report command-line entry point."""

from __future__ import annotations

from spine.bin.report import build_parser, cli


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
