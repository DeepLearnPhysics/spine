"""Reduction and visualization helpers for scalar metric distributions.

The helpers in this module distill the useful plotting patterns from the
full-chain metrics notebook into summary-based functions. Histograms and
quantiles are computed without retaining raw samples, and plotters return
figures so callers retain control over artifact persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .plot import plotting


def histogram_quantiles(
    counts: Sequence[int] | np.ndarray,
    edges: Sequence[float] | np.ndarray,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> list[float | None]:
    """Estimate distribution quantiles from binned sufficient statistics.

    Values are interpolated uniformly within the selected bin. This preserves
    the streaming reduction contract without retaining point-level arrays.
    Empty histograms produce ``None`` for every requested quantile.

    Parameters
    ----------
    counts : sequence of int or np.ndarray
        Number of entries in each histogram bin.
    edges : sequence of float or np.ndarray
        Histogram edges, with one more element than ``counts``.
    quantiles : sequence of float, optional
        Cumulative probabilities to estimate.

    Returns
    -------
    list
        Estimated values in requested order, or ``None`` values for an empty
        histogram.
    """
    counts_array = np.asarray(counts, dtype=np.float64)
    edges_array = np.asarray(edges, dtype=np.float64)
    total = counts_array.sum()
    if total == 0:
        return [None] * len(quantiles)

    cumulative = np.cumsum(counts_array)
    values = []
    for quantile in quantiles:
        target = float(quantile) * total
        index = min(
            int(np.searchsorted(cumulative, target, side="left")), len(counts_array) - 1
        )
        previous = cumulative[index - 1] if index else 0.0
        fraction = (
            (target - previous) / counts_array[index] if counts_array[index] else 0.0
        )
        values.append(
            float(edges_array[index] + fraction * np.diff(edges_array)[index])
        )
    return values


def plot_histogram_with_boxplot(
    distributions: Mapping[str, Mapping[str, Any]],
    edges: Sequence[float],
    *,
    x_label: str,
    yscale: str = "log",
    figsize: tuple[float, float] = (9.0, 6.0),
):
    """Draw notebook-style step histograms with compact box summaries.

    ``distributions`` entries must contain ``histogram`` and may contain
    ``quantiles`` and ``mean``. Quantiles follow the order 10, 25, 50, 75 and
    90 percent. This summary-based interface lets report plots be regenerated
    directly from ``summary.json``.

    Parameters
    ----------
    distributions : mapping
        Legend labels mapped to serialized distribution summaries.
    edges : sequence of float
        Common histogram edges for every distribution.
    x_label : str
        Shared horizontal-axis label.
    yscale : str, default "log"
        Matplotlib scale for the histogram panel. Empty histograms fall back to
        a linear scale to avoid invalid log limits.
    figsize : tuple of float, default (9.0, 6.0)
        Figure width and height in inches.

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel figure containing compact quantiles above step histograms.
    """
    from matplotlib.patches import Rectangle

    plt = plotting()
    edges_array = np.asarray(edges, dtype=np.float64)
    centers = (edges_array[:-1] + edges_array[1:]) / 2.0
    has_entries = any(
        np.any(np.asarray(distribution["histogram"]) > 0)
        for distribution in distributions.values()
    )
    effective_yscale = yscale if has_entries else "linear"

    # The shallow upper panel reproduces the notebook's 10--90 percentile
    # whisker, interquartile box, median line and optional mean diamond.
    with plt.rc_context({"font.size": 14, "figure.autolayout": False}):
        fig, axes = plt.subplots(
            2,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 3]},
        )
        fig.patch.set_alpha(0)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        labels = list(distributions)
        for index, (label, distribution) in enumerate(distributions.items()):
            color = colors[index % len(colors)]
            axes[1].step(
                centers,
                distribution["histogram"],
                where="mid",
                linewidth=2,
                color=color,
                label=label,
            )
            quantiles = distribution.get("quantiles")
            if quantiles and all(value is not None for value in quantiles):
                q10, q25, q50, q75, q90 = quantiles
                y = len(labels) - index - 1
                axes[0].hlines(y, q10, q90, color=color, linewidth=2)
                axes[0].add_patch(
                    Rectangle(
                        (q25, y - 0.25),
                        q75 - q25,
                        0.5,
                        fill=False,
                        edgecolor=color,
                        linewidth=2,
                    )
                )
                axes[0].vlines(q50, y - 0.25, y + 0.25, color=color, linewidth=2)
                mean = distribution.get("mean")
                if mean is not None:
                    axes[0].plot(mean, y, marker="D", markersize=4, color=color)

        axes[0].set_yticks(np.arange(len(labels)), labels=labels[::-1])
        axes[0].grid(True)
        axes[1].set(xlabel=x_label, ylabel="Entries", yscale=effective_yscale)
        axes[1].set_xlim(edges_array[[0, -1]])
        axes[1].grid(True)
        axes[1].legend()
        fig.subplots_adjust(hspace=0.0)
    return fig


__all__ = [
    "histogram_quantiles",
    "plot_histogram_with_boxplot",
]
