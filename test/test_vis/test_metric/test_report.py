"""Tests for streaming metric report recipes and orchestration."""

from __future__ import annotations

import json

import pandas as pd
import yaml

from spine.vis.metric.report import build_report


def _write_inputs(input_dir):
    """Write two small shards for each supported metric analyzer output."""
    for shard in ("a", "b"):
        directory = input_dir / shard
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "index": [0],
                "file_index": [0],
                "count_00": [3],
                "count_01": [1],
                "count_10": [0],
                "count_11": [2],
            }
        ).to_csv(directory / "segment_eval_summary.csv", index=False)
        for direction, distances in (
            ("truth_to_reco", [0.5, 3.0, -1.0]),
            ("reco_to_truth", [1.5, 6.0, -1.0]),
        ):
            pd.DataFrame(
                {
                    "index": [0, 0, 0],
                    "file_index": [0, 0, 0],
                    "dist": distances,
                    "shape": [0, 1, 1],
                    "closest_shape": [0, 0, -1],
                }
            ).to_csv(directory / f"point_eval_{direction}.csv", index=False)
        for level in ("fragment", "particle", "interaction"):
            pd.DataFrame(
                {
                    "index": [0, 1],
                    "file_index": [0, 0],
                    "ari": [0.8, 1.0],
                    "eff": [0.7, 0.9],
                    "pur": [0.6, 0.8],
                }
            ).to_csv(directory / f"cluster_eval_{level}.csv", index=False)


def _write_config(path, formats=("json",)):
    """Write a report configuration used by integration-style tests."""
    config = {
        "strict": True,
        "formats": list(formats),
        "metadata": {"dataset": "unit-test"},
        "metrics": {
            "segmentation": {
                "name": "segment_confusion",
                "source": "**/*segment_eval_summary.csv",
                "class_names": ["shower", "track"],
                "chunksize": 1,
            },
            "ppn": {
                "name": "point_proposal",
                "truth_to_reco": "**/*point_eval_truth_to_reco.csv",
                "reco_to_truth": "**/*point_eval_reco_to_truth.csv",
                "distance_thresholds": [1.0, 2.0, 5.0],
                "chunksize": 1,
            },
            "clustering": {
                "name": "cluster_summary",
                "sources": {
                    "fragment": "**/*cluster_eval_fragment.csv",
                    "particle": "**/*cluster_eval_particle.csv",
                    "interaction": "**/*cluster_eval_interaction.csv",
                },
                "chunksize": 1,
            },
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_build_report_streams_shards_and_writes_shared_summary(tmp_path):
    """All reducers should populate the one JSON source used by renderers."""
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "report"
    config_path = tmp_path / "report.yaml"
    _write_inputs(input_dir)
    _write_config(config_path)

    summary = build_report(config_path, input_dir, output_dir)

    stored = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert stored == summary
    assert summary["inputs"] == {
        "directory": str(input_dir.resolve()),
        "csv_shards": 12,
        "events": 4,
        "data_files": 2,
    }
    segmentation = summary["metrics"]["segmentation"]
    assert segmentation["matrix"] == [[6, 2], [0, 4]]
    assert segmentation["accuracy"] == 10 / 12
    assert (
        summary["metrics"]["ppn"]["directions"]["efficiency"]["threshold_fraction"][
            "1.0"
        ]
        == 2 / 6
    )
    assert (
        summary["metrics"]["clustering"]["levels"]["fragment"]["metrics"]["ari"]["mean"]
        == 0.9
    )


def test_build_report_renders_expected_pngs(tmp_path):
    """Rendering should consume summaries and create the documented artifacts."""
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "report"
    config_path = tmp_path / "report.yaml"
    _write_inputs(input_dir)
    _write_config(config_path, formats=("png", "json"))

    build_report(config_path, input_dir, output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "summary.json",
        "segmentation_confusion.png",
        "ppn_efficiency.png",
        "ppn_purity.png",
        "ppn_resolution.png",
        "clustering_summary.png",
    }


def test_non_strict_report_records_missing_metric(tmp_path):
    """Non-strict reports should describe missing optional metric inputs."""
    config_path = tmp_path / "report.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "strict": False,
                "formats": ["json"],
                "metrics": {
                    "segmentation": {
                        "name": "segment_confusion",
                        "source": "**/missing.csv",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_report(config_path, tmp_path / "raw", tmp_path / "report")

    assert summary["metrics"]["segmentation"]["status"] == "skipped"
