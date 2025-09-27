"""Visualization tools for confusion matrices."""

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mpl.rcParams.update({"font.size": 18})
mpl.rcParams.update({"figure.autolayout": True})


def draw_confusion_matrix(
    file_path,
    num_classes=None,
    mapping=None,
    figure_name="confmat",
    show_counts=False,
    class_names=None,
    figsize=(9, 6),
    norm_axis=0,
):
    """Draws the confusion matrix from a file produced by the analysis
    script used to evaluate the classification accuracy.

    Parameters
    ----------
    file_path : str
        Path to the file which contains the classification metrics
    num_classes : int, optional
        Number of classes to represent
    mapping : dict, optional
        Mapping between the stored class and a redefined set of classes
    show_counts : bool, default False
        Show the number of entries in the contingency matrix
    class_names : list, optional
        Labels for each class
    figsize : tuple, default (9, 6)
        Figure size
    norm_axis : int, default 0
        Normalization axis (0: recall, 1: precision)
    """
    # Load the file
    data = pd.read_csv(file_path)

    # Load the contingency matrix
    if "score_0" in data.keys():
        hist = build_matrix(data, num_classes, mapping)
    else:
        hist = rebuild_matrix(data, num_classes, mapping)

    # Initialize figure
    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0)

    # Normalize the histogram counts to the total number of entries in each true class bin
    assert norm_axis in (
        0,
        1,
    ), "The normalization axis must be 0 (recall) or 1 (precision)."
    norms = np.sum(hist, axis=norm_axis)
    if norm_axis == 0:
        hist_norm = hist / norms
    else:
        hist_norm = hist / norms[:, None]

    # Initialize plot, fill
    num_classes = len(hist)
    xedges = yedges = -0.5 + np.arange(0, num_classes + 1)
    plt.pcolormesh(xedges, yedges, hist_norm, cmap="Blues")
    for i in range(num_classes):
        for j in range(num_classes):
            label = (
                "{:0.3f}\n({})".format(hist_norm[i, j], int(hist[i, j]))
                if show_counts
                else "{:0.3f}".format(hist_norm[i, j])
            )
            plt.text(
                j,
                i,
                label,
                color="white" if hist_norm[i, j] > 0.5 else "black",
                ha="center",
                va="center",
            )

    # Set axes style and labels
    plt.xlabel("Class label")
    plt.ylabel("Class prediction")
    if class_names is not None:
        assert (
            len(class_names) == num_classes
        ), "Must provide one class label per class."
        plt.xticks(np.arange(num_classes), labels=class_names)
        plt.yticks(np.arange(num_classes), labels=class_names)
    plt.colorbar()

    # Save and show
    plt.savefig(f"{figure_name}.png", bbox_inches="tight")
    plt.show()


def build_matrix(data, num_classes=None, mapping=None):
    """Builds a confusion matrix from a pixel-wise storage file.

    Parameters
    ----------
    data : pd.Dataframe
        Dataframe which contains the pixel label/predictions
    num_classes : int, optional
        Number of classes to represent
    mapping : dict, optional
        Mapping between the stored class and a redefined set of classes
    """
    # If the number of classes is not specified, fetch from the file
    if num_classes is None:
        if mapping is not None:
            num_classes = len(mapping)
        else:
            classes = []
            for k in data.keys():
                if k.startswith("score"):
                    classes.append(int(k[-1]))
            num_classes = np.max(classes) + 1

    assert (
        mapping is None or len(mapping) == num_classes
    ), "The number of classes should match those in the map."

    # Build the confusion matrix
    hist = np.histogram2d(
        data.pred,
        data.label,
        bins=[num_classes, num_classes],
        range=[[0, num_classes], [0, num_classes]],
    )[0]

    return hist


def rebuild_matrix(data, num_classes=None, mapping=None):
    """Builds a confusion matrix from an entry-wise storage file.

    Parameters
    ----------
    data : pd.Dataframe
        Dataframe which contains the flattened matrix
    num_classes : int, optional
        Number of classes to represent
    mapping : dict, optional
        Mapping between the stored class and a redefined set of classes
    """
    # If the number of classes is not specified, fetch from the file
    if num_classes is None:
        if mapping is not None:
            num_classes = len(mapping)
        else:
            classes = []
            for k in data.keys():
                if k.startswith("count"):
                    classes.append(int(k[-1]))
            num_classes = np.max(classes) + 1

    assert (
        mapping is None or len(mapping) == num_classes
    ), "The number of classes should match those in the map."

    # Rebuild confusion matrix
    hist = np.empty((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            if mapping is None or not (i in mapping and j in mapping):
                hist[i, j] = np.sum(data[f"count_{i}{j}"])
            else:
                hist[i, j] = 0
                for k in mapping[i]:
                    for l in mapping[j]:
                        hist[i, j] += np.sum(data[f"count_{k}{l}"])

    return hist
