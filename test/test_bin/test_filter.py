"""Tests for the standalone ``spine-filter`` command."""

from spine.bin.filter import build_parser, cli


def test_filter_parser_exposes_scan_and_build_contracts():
    """Both scheduler-friendly phases should accept source-list inputs."""
    parser = build_parser()
    scan = parser.parse_args(
        [
            "scan",
            "--config",
            "filter.yaml",
            "--source-list",
            "files.txt",
            "--cache-dir",
            "cache",
            "--workers",
            "3",
            "--force",
        ]
    )
    build = parser.parse_args(
        [
            "build",
            "--config",
            "filter.yaml",
            "--source",
            "one.root",
            "two.root",
            "--cache-dir",
            "cache",
            "--output",
            "filter-out.yaml",
            "--output-source-list",
            "filtered.txt",
        ]
    )

    assert scan.command == "scan"
    assert scan.source_list == "files.txt"
    assert scan.workers == 3
    assert scan.force
    assert build.command == "build"
    assert build.source == ["one.root", "two.root"]


def test_filter_cli_dispatches_scan(monkeypatch, capsys):
    """Scan should forward reusable API arguments and summarize cache use."""
    calls = []

    def scan_sources(**kwargs):
        calls.append(kwargs)
        return [
            {"source": "one", "record": "one.yaml", "reused": True},
            {"source": "two", "record": "two.yaml", "reused": False},
        ]

    monkeypatch.setattr("spine.bin.filter.scan_sources", scan_sources)
    status = cli(
        [
            "scan",
            "--config",
            "filter.yaml",
            "--source-list",
            "files.txt",
            "--cache-dir",
            "cache",
        ]
    )

    assert status == 0
    assert calls[0]["source_list"] == "files.txt"
    assert calls[0]["workers"] == 1
    assert "1 reused, 1 scanned" in capsys.readouterr().out


def test_filter_cli_dispatches_build(monkeypatch, capsys):
    """Build should forward output paths and report acceptance totals."""
    calls = []

    def build_manifest(**kwargs):
        calls.append(kwargs)
        return {"accepted_entries": 8, "rejected_entries": 2}

    monkeypatch.setattr("spine.bin.filter.build_manifest", build_manifest)
    status = cli(
        [
            "build",
            "--config",
            "filter.yaml",
            "--source",
            "one.root",
            "--cache-dir",
            "cache",
            "--output",
            "out.yaml",
            "--output-source-list",
            "files.txt",
        ]
    )

    assert status == 0
    assert calls[0]["output"] == "out.yaml"
    assert calls[0]["output_source_list"] == "files.txt"
    assert "8 accepted, 2 rejected" in capsys.readouterr().out
