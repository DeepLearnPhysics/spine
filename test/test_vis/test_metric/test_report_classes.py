"""Tests for constant-derived report classes and configurable aggregation."""

from __future__ import annotations

import pandas as pd

from spine.vis.metric.report import SegmentConfusionRecipe
from spine.vis.metric.report.classification import resolve_class_groups


def test_class_groups_use_constant_labels_and_allow_restriction():
    """Canonical names should resolve without duplicating display labels."""
    groups = resolve_class_groups(
        {"classes": ["shower", "track", "low_energy"]},
        kind="shape",
        default_ids=range(5),
    )

    assert groups == [
        {"name": "Shower", "source_ids": [0]},
        {"name": "Track", "source_ids": [1]},
        {"name": "LE", "source_ids": [4]},
    ]


def test_segmentation_can_map_non_ghost_classes_together(tmp_path):
    """Many-to-one mapping should support direct ghost/non-ghost reports."""
    path = tmp_path / "segment_eval_summary.csv"
    row = {
        f"count_{prediction}{truth}": 0 for prediction in range(6) for truth in range(6)
    }
    row.update({"count_00": 4, "count_05": 1, "count_50": 2, "count_55": 3})
    pd.DataFrame([row]).to_csv(path, index=False)
    recipe = SegmentConfusionRecipe(
        "ghost",
        {
            "class_mapping": {
                "Non-ghost": ["shower", "track", "michel", "delta", "lowe"],
                "Ghost": ["ghost"],
            }
        },
    )

    summary = recipe.reduce({"source": [path]})

    assert summary["class_names"] == ["Non-ghost", "Ghost"]
    assert summary["matrix"] == [[4, 1], [2, 3]]
    assert summary["excluded_count"] == 0
