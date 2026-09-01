"""Tests for the standalone spine-report command-line entry point."""

from __future__ import annotations

from spine.bin.report import build_parser


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
