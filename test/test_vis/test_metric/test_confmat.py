"""Tests for confusion-matrix visualization helpers."""

import numpy as np
import pandas as pd
import pytest

from spine.vis.metric.confmat import build_matrix, draw_confusion_matrix, rebuild_matrix


def test_build_matrix_counts_predictions_by_label():
    data = pd.DataFrame({"pred": [0, 1, 1], "label": [0, 0, 1], "score_0": 0})

    hist = build_matrix(data, num_classes=2)

    np.testing.assert_array_equal(hist, [[1, 0], [1, 1]])


def test_build_matrix_applies_class_mapping():
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


def test_confusion_matrix_infers_classes_and_validation_paths(tmp_path, monkeypatch):
    pixel = pd.DataFrame(
        {
            "pred": [0, 1],
            "label": [0, 1],
            "score_0": [0.8, 0.1],
            "score_1": [0.2, 0.9],
        }
    )
    flat = pd.DataFrame(
        {
            "count_00": [1],
            "count_01": [2],
            "count_10": [3],
            "count_11": [4],
        }
    )
    path = tmp_path / "confmat.csv"
    flat_path = tmp_path / "flat.csv"
    pixel.to_csv(path, index=False)
    flat.to_csv(flat_path, index=False)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    inferred = build_matrix(pixel)
    rebuilt = rebuild_matrix(flat)
    draw_confusion_matrix(
        flat_path,
        figure_name=str(tmp_path / "flat"),
        norm_axis=1,
        show_counts=True,
        class_names=["a", "b"],
    )

    assert inferred.shape == (2, 2)
    assert rebuilt.tolist() == [[1, 2], [3, 4]]
    assert (tmp_path / "flat.png").exists()

    with pytest.raises(ValueError, match="number of classes"):
        build_matrix(pixel, num_classes=3, mapping={0: [0], 1: [1]})
    with pytest.raises(ValueError, match="number of classes"):
        rebuild_matrix(flat, num_classes=3, mapping={0: [0], 1: [1]})
    with pytest.raises(ValueError, match="normalization axis"):
        draw_confusion_matrix(path, num_classes=2, norm_axis=2)
    with pytest.raises(ValueError, match="one class label"):
        draw_confusion_matrix(path, num_classes=2, class_names=["a"])


def test_confusion_matrix_rejects_files_without_inferable_classes():
    pixel = pd.DataFrame({"pred": [0], "label": [0]})
    flat = pd.DataFrame({"foo": [1]})

    with pytest.raises(ValueError, match="Could not infer the number of classes"):
        build_matrix(pixel)
    with pytest.raises(ValueError, match="Could not infer the number of classes"):
        rebuild_matrix(flat)
