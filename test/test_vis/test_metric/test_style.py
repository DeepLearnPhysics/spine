"""Tests for reusable metric plotting styles."""

from __future__ import annotations

from spine.vis.metric.style import histogram_quantiles, plot_histogram_with_boxplot


def test_histogram_quantiles_are_serializable_and_monotonic():
    """Binned quantiles should preserve ordering without raw samples."""
    quantiles = histogram_quantiles([1, 2, 1], [0.0, 1.0, 2.0, 3.0])

    assert all(isinstance(value, float) for value in quantiles)
    assert quantiles == sorted(quantiles)


def test_histogram_with_boxplot_returns_notebook_style_figure():
    """The promoted notebook helper should return a two-axis figure."""
    figure = plot_histogram_with_boxplot(
        {
            "Sample": {
                "histogram": [1, 2, 1],
                "quantiles": [0.4, 1.0, 1.5, 2.0, 2.6],
                "mean": 1.5,
            }
        },
        [0.0, 1.0, 2.0, 3.0],
        x_label="Metric",
    )

    assert len(figure.axes) == 2
