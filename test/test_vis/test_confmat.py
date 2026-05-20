"""Tests for confusion-matrix visualization helpers."""

import numpy as np
import pandas as pd

from spine.vis.metric.confmat import build_matrix, draw_confusion_matrix, rebuild_matrix


def test_build_matrix_counts_predictions_by_label():
    """Pixel-wise confusion matrices should count prediction/label pairs."""
    data = pd.DataFrame({"pred": [0, 1, 1], "label": [0, 0, 1], "score_0": 0})

    hist = build_matrix(data, num_classes=2)

    np.testing.assert_array_equal(hist, [[1, 0], [1, 1]])


def test_build_matrix_applies_class_mapping():
    """Pixel-wise confusion matrices should honor requested class regrouping."""
    data = pd.DataFrame(
        {
            "pred": [0, 1, 2, 2],
            "label": [1, 0, 2, 0],
            "score_0": 0,
            "score_1": 0,
            "score_2": 0,
        }
    )
    mapping = {0: [0, 1], 1: [2]}

    hist = build_matrix(data, mapping=mapping)

    np.testing.assert_array_equal(hist, [[2, 0], [1, 1]])


def test_rebuild_matrix_applies_class_mapping():
    """Entry-wise confusion matrices should aggregate mapped source classes."""
    data = pd.DataFrame(
        {
            "count_00": [1],
            "count_01": [2],
            "count_10": [3],
            "count_11": [4],
            "count_02": [5],
            "count_20": [6],
            "count_12": [7],
            "count_21": [8],
            "count_22": [9],
        }
    )
    mapping = {0: [0, 1], 1: [2]}

    hist = rebuild_matrix(data, mapping=mapping)

    np.testing.assert_array_equal(hist, [[10, 12], [14, 9]])


def test_draw_confusion_matrix_writes_figure(tmp_path, monkeypatch):
    """Confusion-matrix drawing should load CSV inputs and save a PNG."""
    data_path = tmp_path / "metrics.csv"
    figure_path = tmp_path / "confmat"
    pd.DataFrame({"pred": [0, 1], "label": [0, 1], "score_0": 0}).to_csv(
        data_path, index=False
    )
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    draw_confusion_matrix(
        data_path,
        num_classes=2,
        figure_name=str(figure_path),
        class_names=["a", "b"],
        show_counts=True,
    )

    assert figure_path.with_suffix(".png").exists()
