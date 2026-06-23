"""Heatmap helpers for reconstruction and analysis metrics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.text import Text
from matplotlib.ticker import Formatter

__all__ = ["heatmap", "annotate_heatmap"]


class UncertaintyFormatter(Formatter):
    """Format heatmap annotations with optional uncertainty information.

    The format string follows :meth:`str.format`. The central value is exposed
    as ``x``, the tick position as ``pos``, and the uncertainty as ``unc``.
    """

    def __init__(self, fmt: str) -> None:
        """Store the annotation format string.

        Parameters
        ----------
        fmt : str
            Format string used to render annotation values.
        """
        self.fmt = fmt

    def __call__(self, x: float | tuple[float, float], pos: int | None = None) -> str:
        """Return one formatted annotation label.

        Parameters
        ----------
        x : Union[float, Tuple[float, float]]
            Central value, or ``(value, uncertainty)`` tuple.
        pos : int, optional
            Tick position forwarded to the format string.

        Returns
        -------
        str
            Formatted label string.
        """
        unc = None
        if isinstance(x, tuple):
            x, unc = x

        return self.fmt.format(x=x, pos=pos, unc=unc)


def heatmap(
    data: np.ndarray,
    row_labels: list[str] | np.ndarray,
    col_labels: list[str] | np.ndarray,
    ax: Axes | None = None,
    **kwargs: Any,
) -> AxesImage:
    """Create a heatmap from a 2D array and row/column labels.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional array of shape ``(N, M)``.
    row_labels : Union[List[str], np.ndarray]
        Labels for the ``N`` heatmap rows.
    col_labels : Union[List[str], np.ndarray]
        Labels for the ``M`` heatmap columns.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the heatmap. If omitted, use the current axes.
    **kwargs : Any
        Additional keyword arguments forwarded to :meth:`Axes.imshow`.

    Returns
    -------
    matplotlib.image.AxesImage
        Heatmap image handle.
    """

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=18)
    ax.set_yticklabels(row_labels, fontsize=18)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(
    im: AxesImage,
    data: np.ndarray | None = None,
    unc: np.ndarray | None = None,
    valfmt: str | Formatter = "{x:.2f}",
    textcolors: tuple[str, str] = ("black", "white"),
    threshold: float | None = None,
    **textkw: Any,
) -> list[Text]:
    """Annotate a heatmap image with formatted cell values.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        Heatmap image to annotate.
    data : np.ndarray, optional
        Data used to annotate the heatmap. If omitted, use ``im.get_array()``.
    unc : np.ndarray, optional
        Uncertainty array matched to ``data``. If omitted, fill with ``NaN``.
    valfmt : Union[str, matplotlib.ticker.Formatter], default ``"{x:.2f}"``
        Annotation formatter. Strings are wrapped in :class:`UncertaintyFormatter`.
    textcolors : Tuple[str, str], default ``("black", "white")``
        Text colors to use below and above the threshold, respectively.
    threshold : float, optional
        Data threshold used to switch between the two text colors. If omitted,
        use the midpoint of the image normalization range.
    **textkw : Any
        Additional keyword arguments forwarded to :meth:`Axes.text`.

    Returns
    -------
    List[matplotlib.text.Text]
        Text artists created for each heatmap cell.
    """

    # Fetch data and uncertainty arrays if they are not provided explicitly.
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    if unc is None:
        unc = np.full_like(data, np.nan, dtype=float)

    assert data is not None  # for the type checker
    assert unc is not None  # for the type checker

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = {"horizontalalignment": "center", "verticalalignment": "center"}
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = UncertaintyFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if isinstance(valfmt, UncertaintyFormatter):
                label = valfmt((data[i, j], unc[i, j]), None)
            else:
                label = valfmt(data[i, j], None)
            text = im.axes.text(j, i, label, None, **kw)
            texts.append(text)

    return texts
