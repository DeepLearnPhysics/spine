"""Tests for the shipped generic full-chain metric configurations."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parents[3] / "config/test"


def test_generic_metric_configs_are_flat():
    """Keep generic reporting defaults directly under ``config/test``."""
    assert {path.name for path in CONFIG_DIR.glob("*.yaml")} == {
        "analyzers.yaml",
        "report.yaml",
    }
    assert not any(path.is_dir() for path in CONFIG_DIR.iterdir())


def test_generic_metric_configs_are_undated_and_ghost_free():
    """SPINE should ship generic defaults without production versioning."""
    analyzer_path = CONFIG_DIR / "analyzers.yaml"
    report_path = CONFIG_DIR / "report.yaml"
    analyzer = yaml.safe_load(analyzer_path.read_text(encoding="utf-8"))
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))

    assert analyzer["ana"]["segment_eval"]["ghost"] is False
    assert "__meta__" not in analyzer
    assert "full_chain_version" not in report["metadata"]
    assert not (CONFIG_DIR / "analyzers_240805.yaml").exists()
    assert not (CONFIG_DIR / "report_240805.yaml").exists()


def test_generic_save_records_expose_node_truth_selections():
    """SaveAna defaults should include fragment and neutrino truth fields."""
    analyzer = yaml.safe_load(
        (CONFIG_DIR / "analyzers.yaml").read_text(encoding="utf-8")
    )
    save = analyzer["ana"]["save"]

    assert "group_primary" in save["fragment"]
    assert "nu_id" in save["particle"]


def test_generic_fragment_metrics_use_adapted_truth_indexes():
    """Full-chain fragment metrics should use populated adapted indexes."""
    analyzer = yaml.safe_load(
        (CONFIG_DIR / "analyzers.yaml").read_text(encoding="utf-8")
    )
    report = yaml.safe_load((CONFIG_DIR / "report.yaml").read_text(encoding="utf-8"))

    assert analyzer["post"]["match"]["truth_point_mode"] == "points_adapt"
    assert analyzer["ana"]["cluster_eval"]["truth_index_mode"] == "index_adapt"
    assert "size_adapt" in analyzer["ana"]["save"]["fragment"]

    cuts = report["metrics"]["nodes"]["tasks"]["shower_fragment_primary"][
        "quality_cuts"
    ]["all"]
    assert {"column": "truth_size_adapt", "min": 25} in cuts


def test_generic_report_includes_standard_and_mapped_semantics():
    """Generic defaults should retain five classes and demonstrate mapping."""
    report = yaml.safe_load((CONFIG_DIR / "report.yaml").read_text(encoding="utf-8"))
    metrics = report["metrics"]

    assert metrics["segmentation"]["classes"] == [
        "shower",
        "track",
        "michel",
        "delta",
        "low_energy",
    ]
    assert metrics["segmentation_shower_track"]["class_mapping"] == {
        "Shower": ["shower", "michel", "delta", "low_energy"],
        "Track": ["track"],
    }


def test_generic_orientation_is_restricted_to_tracks():
    """Particle orientation should only evaluate track-like particles."""
    report = yaml.safe_load((CONFIG_DIR / "report.yaml").read_text(encoding="utf-8"))
    cuts = report["metrics"]["nodes"]["tasks"]["particle_orientation"]["quality_cuts"][
        "all"
    ]

    assert {"column": "truth_shape", "equals": 1} in cuts
    assert not any(cut.get("column") == "truth_shape" and "in" in cut for cut in cuts)
