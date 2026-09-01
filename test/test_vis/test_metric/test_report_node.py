"""Tests for configurable save-record node reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spine.vis.metric.report import NodeSummaryRecipe, quality_cut_mask


def test_quality_cut_mask_supports_nested_notebook_selection():
    """Nested negation should clearly exclude electrons from neutral pions."""
    frame = pd.DataFrame(
        {
            "match_overlap": [0.8, 0.8, 0.8, 0.2],
            "truth_nu_id": [0, 0, -1, 0],
            "truth_is_primary": [True, True, True, True],
            "truth_parent_pdg_code": [0, 111, 111, 0],
            "truth_pdg_code": [11, 11, 22, 13],
        }
    )
    cuts = {
        "all": [
            {"column": "match_overlap", "min": 0.5},
            {"column": "truth_nu_id", "min": 0},
            {"column": "truth_is_primary", "equals": True},
            {
                "not": {
                    "all": [
                        {"column": "truth_parent_pdg_code", "equals": 111},
                        {"column": "truth_pdg_code", "abs_equals": 11},
                    ]
                }
            },
        ]
    }

    assert quality_cut_mask(frame, cuts).tolist() == [True, False, False, False]


def test_node_recipe_reduces_classification_and_orientation(tmp_path):
    """Matched save rows should produce inspectable node task records."""
    path = tmp_path / "sample_save_truth_particles.csv"
    pd.DataFrame(
        {
            "index": [0, 0, 1],
            "file_index": [0, 0, 0],
            "match_overlap": [0.9, 0.8, 0.2],
            "truth_size": [30, 40, 50],
            "truth_is_primary": [1, 0, 1],
            "reco_is_primary": [1, 1, 0],
            "truth_start_dir_x": [1.0, 0.0, 1.0],
            "truth_start_dir_y": [0.0, 1.0, 0.0],
            "truth_start_dir_z": [0.0, 0.0, 0.0],
            "reco_start_dir_x": [1.0, 0.0, -1.0],
            "reco_start_dir_y": [0.0, -1.0, 0.0],
            "reco_start_dir_z": [0.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)
    config = {
        "tasks": {
            "primary": {
                "source": "particle",
                "type": "classification",
                "truth_column": "truth_is_primary",
                "prediction_column": "reco_is_primary",
                "class_names": ["secondary", "primary"],
                "quality_cuts": {
                    "all": [
                        {"column": "match_overlap", "min": 0.5},
                        {"column": "truth_size", "min": 25},
                    ]
                },
            },
            "orientation": {
                "source": "particle",
                "type": "orientation",
                "truth_columns": [
                    "truth_start_dir_x",
                    "truth_start_dir_y",
                    "truth_start_dir_z",
                ],
                "prediction_columns": [
                    "reco_start_dir_x",
                    "reco_start_dir_y",
                    "reco_start_dir_z",
                ],
                "quality_cuts": {"column": "match_overlap", "min": 0.5},
            },
        }
    }

    summary = NodeSummaryRecipe("nodes", config).reduce({"particle": [path]})

    assert summary["tasks"]["primary"]["matrix"] == [[0, 0], [1, 1]]
    assert summary["tasks"]["primary"]["accuracy"] == 0.5
    assert summary["tasks"]["primary"]["evaluated_rows"] == 2
    assert summary["tasks"]["orientation"]["distribution"]["count"] == 2
    assert summary["tasks"]["orientation"]["evaluated_rows"] == 2
    assert summary["tasks"]["orientation"]["forward_fraction"] == 0.5


def test_quality_cut_mask_supports_all_leaf_predicates():
    """Every documented leaf operator and boolean disjunction should compose."""
    frame = pd.DataFrame({"value": [-3, -2, 0, 2, 3], "kind": [0, 1, 2, 3, 4]})

    specification = {
        "any": [
            {
                "all": [
                    {"column": "value", "min": -2, "max": 2},
                    {"column": "value", "not_equals": 0},
                    {"column": "kind", "in": [1, 3]},
                    {"column": "kind", "not_in": [0, 4]},
                ]
            },
            {"column": "value", "abs_equals": 3, "abs_not_equals": 2},
        ]
    }

    assert quality_cut_mask(frame, specification).tolist() == [
        True,
        True,
        False,
        True,
        True,
    ]
    assert quality_cut_mask(frame, None).tolist() == [True] * 5
    with pytest.raises(ValueError, match="missing"):
        quality_cut_mask(frame, {"column": "unknown", "equals": 1})


@pytest.mark.parametrize(
    ("config", "sources", "message"),
    [
        ({}, {}, "non-empty `tasks`"),
        (
            {"tasks": {"primary": {"source": "particle"}}},
            {},
            "undefined sources",
        ),
    ],
)
def test_node_recipe_validates_tasks(config, sources, message):
    """Missing tasks and source mismatches should fail before reading CSVs."""
    with pytest.raises(ValueError, match=message):
        NodeSummaryRecipe("nodes", config).reduce(sources)


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ({"type": "classification"}, "needs `truth_column`"),
        (
            {
                "type": "classification",
                "truth_column": "truth_shape",
                "class_type": "energy",
            },
            "Unknown class type",
        ),
        (
            {"type": "orientation", "range": [0.0]},
            "must contain two bounds",
        ),
        ({"type": "regression"}, "Unknown node task type"),
    ],
)
def test_node_recipe_validates_task_definitions(task, message):
    """Unsupported or incomplete task definitions should fail clearly."""
    with pytest.raises(ValueError, match=message):
        NodeSummaryRecipe._initialize_task("task", task)


def test_node_recipe_initializes_shape_and_pid_domains():
    """Shape and PID tasks should infer their complete canonical class sets."""
    shape = NodeSummaryRecipe._initialize_task("shape", {"truth_column": "truth_shape"})
    pid = NodeSummaryRecipe._initialize_task("pid", {"truth_column": "truth_pid"})

    assert shape["class_names"][:2] == ["Shower", "Track"]
    assert pid["class_names"][:3] == ["Photon", "Electron", "Muon"]


def test_node_recipe_rejects_missing_task_columns():
    """Classification and orientation updates require every configured column."""
    recipe = NodeSummaryRecipe("nodes", {})
    classification = recipe._initialize_task(
        "primary", {"truth_column": "truth_is_primary"}
    )
    with pytest.raises(ValueError, match="classification columns"):
        recipe._update_task(
            classification,
            {
                "truth_column": "truth_is_primary",
                "prediction_column": "reco_is_primary",
            },
            pd.DataFrame({"truth_is_primary": [1]}),
        )

    with pytest.raises(ValueError, match="require string truth and prediction"):
        recipe._update_task(
            classification,
            {"truth_column": "truth_is_primary"},
            pd.DataFrame({"truth_is_primary": [1]}),
        )

    orientation = recipe._initialize_task("orientation", {"type": "orientation"})
    with pytest.raises(ValueError, match="orientation columns"):
        recipe._update_task(
            orientation,
            {
                "truth_columns": ["truth_x"],
                "prediction_columns": ["reco_x"],
            },
            pd.DataFrame({"truth_x": [1.0]}),
        )


def test_node_recipe_handles_empty_orientation_and_renders_tasks(tmp_path):
    """Invalid vectors should serialize safely and both task types should render."""
    path = tmp_path / "particles.csv"
    pd.DataFrame(
        {
            "truth_is_primary": [1],
            "reco_is_primary": [1],
            "truth_x": [0.0],
            "reco_x": [np.nan],
        }
    ).to_csv(path, index=False)
    recipe = NodeSummaryRecipe(
        "nodes",
        {
            "tasks": {
                "primary": {
                    "source": "particle",
                    "truth_column": "truth_is_primary",
                    "prediction_column": "reco_is_primary",
                },
                "orientation": {
                    "source": "particle",
                    "type": "orientation",
                    "truth_columns": ["truth_x"],
                    "prediction_columns": ["reco_x"],
                },
            }
        },
    )

    summary = recipe.reduce({"particle": [path]})
    artifacts = recipe.render(summary, tmp_path, ["png"])

    assert summary["tasks"]["orientation"]["evaluated_rows"] == 0
    assert summary["tasks"]["orientation"]["forward_fraction"] == 0.0
    assert {artifact.name for artifact in artifacts} == {
        "node_primary.png",
        "node_orientation.png",
    }
