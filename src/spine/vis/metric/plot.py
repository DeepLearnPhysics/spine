"""Matplotlib lifecycle helpers shared by metric visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["plotting", "save_figure"]


def plotting():
    """Return ``matplotlib.pyplot`` configured for non-interactive jobs.

    Matplotlib is imported lazily so summary-only consumers do not pay its
    import cost. The ``Agg`` backend also prevents batch workers from requiring
    an X server.

    Returns
    -------
    module
        Configured :mod:`matplotlib.pyplot` module.
    """
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
    list of Path
        Paths of the figure files written by this call.

    Notes
    -----
    The figure is closed after every requested format has been written. A
    caller should therefore finish all modifications before calling this
    helper.
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
