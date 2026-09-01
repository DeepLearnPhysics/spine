"""Tests for confusion-matrix visualization helpers."""

import numpy as np
import pandas as pd
import pytest

from spine.vis.metric.confmat import build_matrix, plot_confusion_matrix


def test_build_matrix_counts_predictions_by_label():
    """Prediction records should populate prediction-by-truth counts."""
    data = pd.DataFrame({"pred": [0, 1, 1], "label": [0, 0, 1]})

    matrix = build_matrix(data, num_classes=2)

    np.testing.assert_array_equal(matrix, [[1, 0], [1, 1]])


def test_build_matrix_applies_class_mapping():
    """Many-to-one mappings should aggregate both matrix axes."""
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

    matrix = build_matrix(data, mapping=mapping)

    np.testing.assert_array_equal(matrix, [[2, 0], [1, 1]])


def test_build_matrix_infers_classes_from_score_columns():
    """Score-column suffixes should define the source class count."""
    data = pd.DataFrame(
        {
            "pred": [0, 1],
            "label": [0, 1],
            "score_0": [0.8, 0.1],
            "score_1": [0.2, 0.9],
        }
    )

    matrix = build_matrix(data)

    np.testing.assert_array_equal(matrix, np.eye(2, dtype=np.int64))


def test_build_matrix_validates_class_configuration():
    """Ambiguous or inconsistent class configurations should fail clearly."""
    data = pd.DataFrame(
        {
            "pred": [0, 1],
            "label": [0, 1],
            "score_0": [0.8, 0.1],
            "score_1": [0.2, 0.9],
        }
    )

    with pytest.raises(ValueError, match="number of classes"):
        build_matrix(data, num_classes=3, mapping={0: [0], 1: [1]})
    with pytest.raises(ValueError, match="Could not infer the number of classes"):
        build_matrix(data[["pred", "label"]])


def test_plot_confusion_matrix_uses_shared_heatmap_annotations():
    """The matrix renderer should return one annotated heatmap figure."""
    figure = plot_confusion_matrix([[3, 1], [2, 4]], ["a", "b"])

    matrix_axis = figure.axes[0]
    labels = [text.get_text() for text in matrix_axis.texts]
    assert len(labels) == 4
    assert "(3)" in labels[0]
    assert matrix_axis.get_xlabel() == "True class"
    assert matrix_axis.get_ylabel() == "Predicted class"


@pytest.mark.parametrize("normalize", ["truth", "prediction"])
def test_plot_confusion_matrix_supports_both_normalizations(normalize):
    """Truth and prediction normalization should both produce finite images."""
    figure = plot_confusion_matrix(
        [[0, 0], [0, 2]],
        ["a", "b"],
        normalize=normalize,
        show_counts=False,
    )

    image = figure.axes[0].images[0]
    assert np.isfinite(image.get_array()).all()


def test_plot_confusion_matrix_validates_inputs():
    """Invalid shapes, labels and normalization modes should be rejected."""
    with pytest.raises(ValueError, match="square"):
        plot_confusion_matrix([[1, 2]], ["a"])
    with pytest.raises(ValueError, match="one class name"):
        plot_confusion_matrix([[1, 0], [0, 1]], ["a"])
    with pytest.raises(ValueError, match="Normalization"):
        plot_confusion_matrix([[1]], ["a"], normalize="event")
