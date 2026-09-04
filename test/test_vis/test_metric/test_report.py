"""Tests for streaming metric report recipes and orchestration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from spine.vis.metric.report import (
    ClusterSummaryRecipe,
    PointProposalRecipe,
    SegmentConfusionRecipe,
    build_report,
)
from spine.vis.metric.report.base import InputCounts


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
        "ppn_efficiency_by_class.png",
        "ppn_purity.png",
        "ppn_purity_by_class.png",
        "ppn_resolution.png",
        "ppn_resolution_efficiency_by_class.png",
        "ppn_resolution_purity_by_class.png",
        "clustering_fragment_ari.png",
        "clustering_fragment_ari_by_class.png",
        "clustering_fragment_eff.png",
        "clustering_fragment_eff_by_class.png",
        "clustering_fragment_pur.png",
        "clustering_fragment_pur_by_class.png",
        "clustering_particle_ari.png",
        "clustering_particle_ari_by_class.png",
        "clustering_particle_eff.png",
        "clustering_particle_eff_by_class.png",
        "clustering_particle_pur.png",
        "clustering_particle_pur_by_class.png",
        "clustering_interaction_ari.png",
        "clustering_interaction_eff.png",
        "clustering_interaction_pur.png",
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


def test_input_counts_prefers_physical_event_identity(tmp_path):
    """Complete run coordinates should take precedence over loader indexes."""
    counts = InputCounts()
    frame = pd.DataFrame(
        {
            "run": [1, 1],
            "subrun": [2, 2],
            "event": [3, 3],
            "file_index": [0, 0],
            "index": [7, 7],
        }
    )

    counts.update(tmp_path / "source.csv", frame)

    assert counts.as_dict([tmp_path / "source.csv"], len(frame)) == {
        "csv_shards": 1,
        "rows": 2,
        "events": 1,
        "data_files": 1,
    }


def test_cluster_recipe_reduces_per_class_columns(tmp_path):
    """Per-shape analyzer columns should feed their class distributions."""
    path = tmp_path / "cluster.csv"
    pd.DataFrame({"ari": [0.5], "ari_0": [0.75]}).to_csv(path, index=False)
    recipe = ClusterSummaryRecipe(
        "clustering", {"metric_names": ["ari"], "classes": ["shower"]}
    )

    summary = recipe.reduce({"fragment": [path]})

    assert summary["levels"]["fragment"]["by_class"]["Shower"]["ari"][
        "mean"
    ] == pytest.approx(0.75)


def test_cluster_recipe_rejects_missing_metric_columns(tmp_path):
    """A malformed cluster shard should identify its absent metric columns."""
    path = tmp_path / "cluster.csv"
    pd.DataFrame({"ari": [0.5]}).to_csv(path, index=False)
    recipe = ClusterSummaryRecipe("clustering", {"metric_names": ["ari", "pur"]})

    with pytest.raises(ValueError, match="Missing clustering columns.*pur"):
        recipe.reduce({"fragment": [path]})


def test_cluster_recipe_rejects_malformed_metric_range(tmp_path):
    """Each clustering histogram range should provide exactly two bounds."""
    recipe = ClusterSummaryRecipe(
        "clustering",
        {"metric_names": ["ari"], "metric_ranges": {"ari": [0.0]}},
    )

    with pytest.raises(ValueError, match="must contain two bounds"):
        recipe.reduce({"fragment": [tmp_path / "unused.csv"]})


def test_point_recipe_validates_thresholds_and_distance_column(tmp_path):
    """PPN reduction should reject invalid thresholds and malformed shards."""
    with pytest.raises(ValueError, match="distance thresholds"):
        PointProposalRecipe("ppn", {"distance_thresholds": []}).reduce({})

    path = tmp_path / "points.csv"
    pd.DataFrame({"shape": [0]}).to_csv(path, index=False)
    recipe = PointProposalRecipe("ppn", {})
    with pytest.raises(ValueError, match="Missing `dist` column"):
        recipe.reduce({"truth_to_reco": [path], "reco_to_truth": [path]})

    pd.DataFrame({"dist": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing `shape` column"):
        recipe.reduce({"truth_to_reco": [path], "reco_to_truth": [path]})


def test_point_recipe_filters_and_reduces_thresholds_by_class(tmp_path):
    """Selected shapes should define aggregate and per-class denominators."""
    path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            "dist": [0.5, 2.0, -1.0, 0.25],
            "shape": [0, 0, 1, 3],
            "closest_shape": [0, 1, -1, 3],
        }
    ).to_csv(path, index=False)
    recipe = PointProposalRecipe(
        "ppn",
        {
            "classes": ["shower", "track", "delta"],
            "overall_classes": ["shower", "track"],
            "distance_thresholds": [1.0, 3.0],
        },
    )

    summary = recipe.reduce({"truth_to_reco": [path], "reco_to_truth": [path]})
    direction = summary["directions"]["efficiency"]

    # The delta row is absent from both aggregate counts and class summaries.
    assert direction["total"] == 3
    assert direction["matched"] == 2
    assert direction["threshold_fraction"] == {"1.0": 1 / 3, "3.0": 2 / 3}
    assert direction["by_class"]["Shower"]["total"] == 2
    assert direction["by_class"]["Shower"]["threshold_fraction"] == {
        "1.0": 0.5,
        "3.0": 1.0,
    }
    assert direction["by_class"]["Track"]["total"] == 1
    assert direction["by_class"]["Track"]["threshold_fraction"] == {
        "1.0": 0.0,
        "3.0": 0.0,
    }
    assert direction["by_class"]["Delta"]["total"] == 1
    assert direction["by_class"]["Delta"]["threshold_fraction"] == {
        "1.0": 1.0,
        "3.0": 1.0,
    }


def test_point_renderer_allows_summaries_without_class_breakdowns(tmp_path):
    """Resolution and threshold plots should not require per-class summaries."""
    distribution = {
        "count": 1,
        "mean": 0.5,
        "std": 0.0,
        "quantiles": [0.5] * 5,
        "histogram": [1],
    }
    direction = {
        "threshold_fraction": {"1.0": 1.0},
        "distribution": distribution,
        "by_class": {},
    }
    summary = {
        "distance_thresholds": [1.0],
        "distance_unit": "cm",
        "histogram_edges": [0.0, 1.0],
        "directions": {"efficiency": direction, "purity": direction},
    }

    artifacts = PointProposalRecipe("ppn", {}).render(summary, tmp_path, ["png"])

    assert {artifact.name for artifact in artifacts} == {
        "ppn_efficiency.png",
        "ppn_purity.png",
        "ppn_resolution.png",
    }


@pytest.mark.parametrize(
    ("frame", "config", "message"),
    [
        (pd.DataFrame({"value": [1]}), {}, "No confusion count columns"),
        (
            pd.DataFrame({"count_00": [1], "count_22": [1]}),
            {"num_classes": 2},
            "contains 3",
        ),
    ],
)
def test_segment_recipe_rejects_malformed_count_shards(
    tmp_path, frame, config, message
):
    """Semantic summaries must expose a matrix compatible with configuration."""
    path = tmp_path / "segment.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        SegmentConfusionRecipe("segment", config).reduce({"source": [path]})
