"""Tests for the shipped generic full-chain metric configurations."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parents[3] / "config/test/generic/full_chain"


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
