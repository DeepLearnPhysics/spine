"""Tests for constant-derived report classes and configurable aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spine.vis.metric.report import SegmentConfusionRecipe
from spine.vis.metric.report.classification import (
    aggregate_confusion,
    class_id,
    infer_class_kind,
    map_class_values,
    resolve_class_groups,
)


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


def test_class_helpers_cover_pid_primary_and_numeric_values():
    """All class domains should support canonical aliases and numeric IDs."""
    assert class_id(np.int64(2), "pid") == 2
    assert class_id("muon", "pid") == 2
    assert infer_class_kind("truth_shape") == "shape"
    assert infer_class_kind("truth_pid") == "pid"
    assert infer_class_kind("truth_is_primary") == "primary"
    assert infer_class_kind("truth_group_primary") == "primary"
    assert resolve_class_groups({}, kind="primary", default_ids=(0, 1)) == [
        {"name": "Secondary", "source_ids": [0]},
        {"name": "Primary", "source_ids": [1]},
    ]

    groups = resolve_class_groups(
        {"classes": ["photon", "muon"], "class_names": ["gamma", "mu"]},
        kind="pid",
        default_ids=range(6),
    )
    assert groups == [
        {"name": "gamma", "source_ids": [0]},
        {"name": "mu", "source_ids": [2]},
    ]
    mapped, valid = map_class_values(np.array([0, 2, 5]), groups)
    assert mapped.tolist() == [0, 1, -1]
    assert valid.tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        (
            {"classes": [0], "class_mapping": {"all": [0]}},
            ValueError,
            "either `classes` or `class_mapping`",
        ),
        ({"class_mapping": {}}, TypeError, "non-empty mapping"),
        ({"class_mapping": {"all": "shower"}}, TypeError, "must be a sequence"),
        ({"classes": "shower"}, TypeError, "must be a sequence"),
        ({"classes": []}, ValueError, "must be non-empty"),
        (
            {"class_mapping": {"a": [0], "b": [0]}},
            ValueError,
            "only one report class group",
        ),
        ({"classes": [-1]}, ValueError, "negative sentinel"),
        (
            {"class_mapping": {"all": [0]}, "class_names": ["renamed"]},
            ValueError,
            "cannot be combined",
        ),
        (
            {"classes": [0, 1], "class_names": ["one"]},
            ValueError,
            "one class name",
        ),
    ],
)
def test_class_group_validation_rejects_ambiguous_config(config, error, message):
    """Invalid selection and mapping forms should fail with actionable errors."""
    with pytest.raises(error, match=message):
        resolve_class_groups(config, kind="shape", default_ids=range(5))


def test_class_helpers_reject_unknown_domains_and_out_of_range_groups():
    """Unknown names, columns and matrix indexes should not be accepted."""
    with pytest.raises(ValueError, match="Unknown report class kind"):
        resolve_class_groups({}, kind="energy", default_ids=(0,))
    with pytest.raises(ValueError, match="Unknown shape class"):
        class_id("not-a-shape", "shape")
    with pytest.raises(ValueError, match="Cannot infer a class kind"):
        infer_class_kind("truth_energy")
    with pytest.raises(ValueError, match="exceeds confusion matrix size"):
        aggregate_confusion(
            np.eye(2, dtype=np.int64),
            [{"name": "outside", "source_ids": [2]}],
        )
