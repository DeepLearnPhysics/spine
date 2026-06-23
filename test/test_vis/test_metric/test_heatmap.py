"""Tests for evaluation visualization helpers."""

import matplotlib
import numpy as np
from matplotlib.ticker import Formatter

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from spine.vis.metric.heatmap import annotate_heatmap, heatmap


def test_heatmap_and_annotation_helpers():
    _, ax = plt.subplots()
    image = heatmap(np.array([[0.1, 0.9]]), ["r"], ["a", "b"], ax=ax)
    image_default_ax = heatmap(np.array([[0.1]]), ["r"], ["a"])
    texts = annotate_heatmap(image)
    texts_threshold = annotate_heatmap(image, threshold=0.5)
    texts_unc = annotate_heatmap(
        image,
        unc=np.array([[0.01, 0.02]]),
        valfmt="{x:.1f} +/- {unc:.2f}",
    )

    assert len(texts) == 2
    assert len(texts_threshold) == 2
    assert image_default_ax.axes is not None
    assert texts_unc[0].get_text() == "0.1 +/- 0.01"
    plt.close("all")


def test_annotate_heatmap_supports_generic_formatter_branch():
    class OneDecimalFormatter(Formatter):
        def __call__(self, x, pos=None):
            return f"{x:.1f}"

    _, ax = plt.subplots()
    image = heatmap(np.array([[0.1]]), ["r"], ["a"], ax=ax)
    texts = annotate_heatmap(image, valfmt=OneDecimalFormatter())

    assert texts[0].get_text() == "0.1"
    plt.close("all")
