"""Validation and provenance tests for metric report orchestration."""

from __future__ import annotations

import hashlib

import pytest
import yaml

from spine.vis.metric.report import build_report
from spine.vis.metric.report.manager import _nested_input_counts


def _write_config(path, config):
    """Serialize one test report configuration."""
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_report_hashes_string_checkpoint_metadata(tmp_path):
    """A string checkpoint should normalize to a checksummed metadata map."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    config_path = tmp_path / "report.yaml"
    _write_config(
        config_path,
        {
            "strict": False,
            "formats": ["json"],
            "metadata": {"checkpoint": str(checkpoint)},
            "metrics": {
                "missing": {"name": "segment_confusion", "source": "missing.csv"}
            },
        },
    )

    summary = build_report(config_path, tmp_path / "raw", tmp_path / "output")

    assert summary["metadata"]["checkpoint"] == {
        "path": str(checkpoint),
        "sha256": hashlib.sha256(b"weights").hexdigest(),
    }


def test_report_preserves_configured_checkpoint_checksum(tmp_path):
    """An explicit checksum should remain authoritative without a local file."""
    config_path = tmp_path / "report.yaml"
    _write_config(
        config_path,
        {
            "strict": False,
            "formats": ["json"],
            "metadata": {"checkpoint": {"path": "remote.ckpt", "sha256": "provided"}},
            "metrics": {
                "missing": {"name": "segment_confusion", "source": "missing.csv"}
            },
        },
    )

    summary = build_report(config_path, tmp_path / "raw", tmp_path / "output")

    assert summary["metadata"]["checkpoint"]["sha256"] == "provided"


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({"metrics": {}}, ValueError, "non-empty `metrics`"),
        (
            {"formats": ["jpeg"], "metrics": {"metric": {}}},
            ValueError,
            "Unsupported report formats",
        ),
        ({"metrics": {"metric": []}}, TypeError, "must be a mapping"),
        (
            {"metrics": {"metric": {"name": "unknown", "source": "*.csv"}}},
            ValueError,
            "Unknown report recipe",
        ),
        (
            {"metrics": {"metric": {"name": "cluster_summary", "sources": ["*.csv"]}}},
            TypeError,
            "`sources` must map",
        ),
        (
            {"metrics": {"metric": {"name": "segment_confusion"}}},
            ValueError,
            "must define `source`",
        ),
    ],
)
def test_report_rejects_invalid_configuration(tmp_path, config, error, message):
    """Invalid report structures should fail before reduction begins."""
    config_path = tmp_path / "report.yaml"
    _write_config(config_path, config)

    with pytest.raises(error, match=message):
        build_report(config_path, tmp_path / "raw", tmp_path / "output")


def test_report_rejects_non_mapping_yaml_root(tmp_path):
    """The report YAML root must be a mapping."""
    config_path = tmp_path / "report.yaml"
    _write_config(config_path, ["not", "a", "mapping"])

    with pytest.raises(TypeError, match="must be a mapping"):
        build_report(config_path, tmp_path / "raw", tmp_path / "output")


def test_strict_report_rejects_missing_sources(tmp_path):
    """Strict discovery should identify required inputs which matched nothing."""
    config_path = tmp_path / "report.yaml"
    _write_config(
        config_path,
        {
            "metrics": {
                "segmentation": {
                    "name": "segment_confusion",
                    "source": "missing.csv",
                }
            }
        },
    )

    with pytest.raises(FileNotFoundError, match="found no CSV files"):
        build_report(config_path, tmp_path / "raw", tmp_path / "output")


def test_nested_input_counts_supports_node_source_layout():
    """Top-level counts should understand node summaries grouped by source."""
    metric = {
        "recipe": "node_summary",
        "inputs": {"fragment": {"events": 3}, "particle": {"events": 4}},
    }

    assert _nested_input_counts(metric, "events") == [3, 4]
