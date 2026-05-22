"""Matplotlib and seaborn styling helpers for SPINE visualizations."""

from __future__ import annotations

import matplotlib as mpl
import seaborn as sns

__all__ = ["apply_latex_style", "set_latex_size"]


def apply_latex_style() -> None:
    """Configure Matplotlib and seaborn for compact LaTeX-style figures.

    Returns
    -------
    None
        The styling is applied globally through Matplotlib and seaborn state.
    """
    sns.set_theme(
        rc={
            "figure.figsize": set_latex_size(250),
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 8,
            "font.size": 8,
            "legend.fontsize": 8,
            "legend.labelspacing": 0.25,
            "legend.columnspacing": 0.25,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
        context="paper",
    )
    sns.set_style("white")
    sns.set_style(rc={"axes.grid": True, "font.family": "serif"})
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm}"


def set_latex_size(width: float, fraction: float = 1) -> tuple[float, float]:
    """Compute a figure size in inches from a LaTeX text width.

    Parameters
    ----------
    width : float
        Page width in points.
    fraction : float, default 1
        Fraction of the page width used by the figure.

    Returns
    -------
    Tuple[float, float]
        Figure width and height in inches.
    """
    # Convert from LaTeX points to inches while preserving a visually balanced
    # aspect ratio for figures embedded in papers or notes.
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in
