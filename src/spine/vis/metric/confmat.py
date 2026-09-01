"""Construction and visualization of confusion matrices.

This module has one responsibility: turning an existing set of categorical
predictions into a confusion matrix and drawing a matrix already reduced by an
analyzer or report recipe. CSV discovery, aggregation and artifact persistence
belong to :mod:`spine.vis.metric.report`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .heatmap import annotate_heatmap, heatmap
from .plot import plotting

__all__ = ["build_matrix", "plot_confusion_matrix"]


def build_matrix(
    data: pd.DataFrame,
    num_classes: int | None = None,
    mapping: Mapping[int, Sequence[int]] | None = None,
) -> np.ndarray:
    """Build a confusion matrix from categorical prediction records.

    The input must contain one ``pred`` and ``label`` value per record. Matrix
    rows represent predicted classes and columns represent true classes.

    Parameters
    ----------
    data : pd.DataFrame
        Records containing ``pred`` and ``label`` columns. ``score_<id>``
        columns may be used to infer the source class count.
    num_classes : int, optional
        Number of output classes. If omitted, infer it from ``mapping`` or the
        available score columns.
    mapping : mapping of int to sequence of int, optional
        Output class IDs mapped to source class IDs. Records outside all
        mapped groups are excluded.

    Returns
    -------
    np.ndarray
        Integer confusion matrix with prediction on axis 0 and truth on axis 1.

    Raises
    ------
    ValueError
        If the number of classes cannot be inferred or does not agree with the
        mapping.
    """
    if num_classes is None:
        if mapping is not None:
            num_classes = len(mapping)
        else:
            classes = [
                int(column.rsplit("_", 1)[-1])
                for column in data.columns
                if column.startswith("score_")
            ]
            if not classes:
                raise ValueError(
                    "Could not infer the number of classes from the file. "
                    "Please provide the `num_classes` parameter."
                )
            num_classes = max(classes) + 1

    if mapping is not None and len(mapping) != num_classes:
        raise ValueError("The number of classes should match those in the map.")

    prediction = data["pred"].to_numpy()
    truth = data["label"].to_numpy()
    if mapping is not None:
        mapped_prediction = np.full(len(prediction), -1, dtype=np.int64)
        mapped_truth = np.full(len(truth), -1, dtype=np.int64)
        for class_id, source_ids in mapping.items():
            mapped_prediction[np.isin(prediction, source_ids)] = class_id
            mapped_truth[np.isin(truth, source_ids)] = class_id

        # Mapping acts as both aggregation and selection: a record contributes
        # only when its prediction and truth both belong to configured groups.
        valid = (mapped_prediction >= 0) & (mapped_truth >= 0)
        prediction = mapped_prediction[valid]
        truth = mapped_truth[valid]

    matrix = np.histogram2d(
        prediction,
        truth,
        bins=(num_classes, num_classes),
        range=((0, num_classes), (0, num_classes)),
    )[0]
    return matrix.astype(np.int64)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    *,
    normalize: str = "truth",
    show_counts: bool = True,
    figsize: tuple[float, float] = (9.0, 7.0),
):
    """Draw a confusion matrix with normalized and raw annotations.

    Matrices follow the SPINE analyzer convention: rows are predicted classes
    and columns are true classes. Drawing delegates the labeled grid and cell
    annotations to :mod:`spine.vis.metric.heatmap` so heatmap behavior has a
    single implementation.

    Parameters
    ----------
    matrix : sequence of sequence of int
        Square confusion-count matrix.
    class_names : sequence of str
        Tick label for each matrix class.
    normalize : {"truth", "prediction"}, default "truth"
        Normalize each truth column or prediction row, respectively.
    show_counts : bool, default True
        Include raw counts below normalized values in each cell.
    figsize : tuple of float, default (9.0, 7.0)
        Figure width and height in inches.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the annotated confusion matrix.

    Raises
    ------
    ValueError
        If the matrix is not square, labels do not match its size, or the
        normalization mode is unsupported.
    """
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
        figure, axis = plt.subplots(figsize=figsize)
        figure.patch.set_alpha(0)
        image = heatmap(
            normalized,
            class_names,
            class_names,
            ax=axis,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
        )
        value_format = "{x:.3f}\n({unc:.0f})" if show_counts else "{x:.3f}"
        annotate_heatmap(image, unc=counts, valfmt=value_format, threshold=0.5)
        axis.set(xlabel="True class", ylabel="Predicted class")
        figure.colorbar(
            image,
            ax=axis,
            label=f"{normalize.capitalize()}-normalized fraction",
        )
    return figure
