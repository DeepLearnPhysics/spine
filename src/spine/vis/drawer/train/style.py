"""Style helpers for training-history visualizations."""

from __future__ import annotations

import seaborn as sns
from plotly import graph_objs as go

from ...layout import apply_latex_style

__all__ = ["initialize_matplotlib_style", "initialize_plotly_layout"]


def initialize_plotly_layout() -> go.Layout:
    """Build the default Plotly layout used by ``TrainDrawer``.

    Returns
    -------
    go.Layout
        Plotly layout configured for training and validation curves.
    """
    font = {"size": 20}
    axis_base = {"tickfont": font, "linecolor": "black", "mirror": True}
    return go.Layout(
        template="plotly_white",
        width=1000,
        height=500,
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
        xaxis={"title": {"text": "Epochs", "font": font}, **axis_base},
        yaxis={"title": {"text": "Metric", "font": font}, **axis_base},
        legend={"font": font, "tracegroupgap": 1},
    )


def initialize_matplotlib_style(paper: bool) -> tuple[float, float]:
    """Configure matplotlib and return the linewidth and marker size.

    Parameters
    ----------
    paper : bool
        If ``True``, configure a compact LaTeX-style figure theme.

    Returns
    -------
    Tuple[float, float]
        Line width and marker size to use in Matplotlib plots.
    """
    if paper:
        apply_latex_style()
        return 0.5, 1

    # Use a notebook-oriented seaborn style when not formatting for paper.
    sns.set(rc={"figure.figsize": (9, 6)}, context="notebook", font_scale=2)
    sns.set_style("white")
    sns.set_style(rc={"axes.grid": True})
    return 2, 10
