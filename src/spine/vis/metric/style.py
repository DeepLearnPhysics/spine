"""Reusable Matplotlib styles for reconstruction performance figures.

The helpers in this module distill the useful plotting patterns from the
full-chain metrics notebook into functions which return figures instead of
showing or saving them. Callers therefore retain control over file formats,
paths, and interactive display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def plotting():
    """Return ``matplotlib.pyplot`` configured for non-interactive jobs."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def save_figure(fig: Any, output: Path, formats: Sequence[str]) -> list[Path]:
    """Save a figure in each requested graphical format and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to persist.
    output : Path
        Output path without a suffix.
    formats : sequence of str
        Requested report formats. ``json`` is ignored because figures are
        represented in the report summary separately.

    Returns
    -------
    list[Path]
        Paths of the figure files written by this call.
    """
    plt = plotting()
    paths = []
    for file_format in formats:
        if file_format == "json":
            continue
        path = output.with_suffix(f".{file_format}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def histogram_quantiles(
    counts: Sequence[int],
    edges: Sequence[float],
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> list[float | None]:
    """Estimate distribution quantiles from binned sufficient statistics.

    Values are interpolated uniformly within the selected bin. This preserves
    the streaming reduction contract without retaining point-level arrays.
    Empty histograms produce ``None`` for every requested quantile.
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


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    *,
    normalize: str = "truth",
    show_counts: bool = True,
    figsize: tuple[float, float] = (9.0, 7.0),
):
    """Draw a count matrix with truth- or prediction-normalized annotations."""
    plt = plotting()
    counts = np.asarray(matrix, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("A confusion matrix must be square.")
    if len(class_names) != len(counts):
        raise ValueError("Must provide one class name per confusion-matrix class.")
    if normalize not in ("truth", "prediction"):
        raise ValueError("Normalization must be `truth` or `prediction`.")

    denominator = counts.sum(axis=0) if normalize == "truth" else counts.sum(axis=1)
    if normalize == "prediction":
        denominator = denominator[:, None]
    normalized = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts, denominator, out=normalized, where=denominator != 0)

    with plt.rc_context({"font.size": 14, "figure.autolayout": True}):
        fig, axis = plt.subplots(figsize=figsize)
        fig.patch.set_alpha(0)
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for prediction in range(len(counts)):
            for truth in range(len(counts)):
                value = normalized[prediction, truth]
                label = f"{value:.3f}"
                if show_counts:
                    label += f"\n({counts[prediction, truth]:d})"
                axis.text(
                    truth,
                    prediction,
                    label,
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                )
        axis.set(
            xlabel="True class",
            ylabel="Predicted class",
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
        )
        fig.colorbar(
            image, ax=axis, label=f"{normalize.capitalize()}-normalized fraction"
        )
    return fig


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
    """
    plt = plotting()
    edges_array = np.asarray(edges, dtype=np.float64)
    centers = (edges_array[:-1] + edges_array[1:]) / 2.0
    has_entries = any(
        np.any(np.asarray(distribution["histogram"]) > 0)
        for distribution in distributions.values()
    )
    effective_yscale = yscale if has_entries else "linear"

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
                    plt.Rectangle(
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
    "plot_confusion_matrix",
    "plot_histogram_with_boxplot",
    "plotting",
    "save_figure",
]
