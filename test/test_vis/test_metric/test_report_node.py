"""Tests for configurable save-record node reporting."""

from __future__ import annotations

import pandas as pd

from spine.vis.metric.report import NodeSummaryRecipe, quality_cut_mask


def test_quality_cut_mask_supports_nested_notebook_selection():
    """Nested predicates should express the notebook's primary PID cuts."""
    frame = pd.DataFrame(
        {
            "match_overlap": [0.8, 0.8, 0.2],
            "truth_is_primary": [True, True, True],
            "truth_parent_pdg_code": [0, 111, 0],
            "truth_pdg_code": [11, 11, 13],
        }
    )
    cuts = {
        "all": [
            {"column": "match_overlap", "min": 0.5},
            {"column": "truth_is_primary", "equals": True},
            {
                "any": [
                    {"column": "truth_parent_pdg_code", "not_equals": 111},
                    {"column": "truth_pdg_code", "abs_not_equals": 11},
                ]
            },
        ]
    }

    assert quality_cut_mask(frame, cuts).tolist() == [True, False, False]


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
    assert summary["tasks"]["orientation"]["distribution"]["count"] == 2
    assert summary["tasks"]["orientation"]["forward_fraction"] == 0.5
