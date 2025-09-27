"""Various metrics used to evaluate clustering."""

import numba as nb
import numpy as np

from spine.math.linalg import contingency_table
from spine.math.metrics import adjusted_mutual_info_score, adjusted_rand_score

__all__ = ["pur", "eff", "pur_eff", "ari", "ami", "sbd"]


def pur(truth, pred, batch_ids=None, per_cluster=True):
    """Assignment purity.

    Parameters
    ----------
    truth : np.ndarray
        (N) Set of true labels
    pred : np.ndarray
        (N) Set of predicted labels
    batch_ids : np.ndarray, optional
        (N) Batch IDs
    per_cluster : bool, default True
        If `True`, computes the purity per predicted cluster, than averages it

    Returns
    -------
    float
        Assignment purity
    """
    # If the vectors compared are empty, nothing to do
    if len(truth) == 0:
        return -1.0

    # Transform labels to be unique across all batch entries
    truth, _, truth_counts = unique_labels(truth, batch_ids)
    pred, _, pred_counts = unique_labels(pred, batch_ids)

    # Compute the contingency table
    table = contingency_table(truth, pred, len(truth_counts), len(pred_counts))

    # Evaluate the purity for each predicted cluster
    if per_cluster:
        purities = table.max(axis=0) / pred_counts
        return purities.mean()

    else:
        purity = np.sum(table.max(axis=0)) / len(pred)
        return purity


def eff(truth, pred, batch_ids=None, per_cluster=True):
    """Assignment efficiency, evaluated per true cluster and averaged.

    Parameters
    ----------
    truth : np.ndarray
        (N) Set of true labels
    pred : np.ndarray
        (N) Set of predicted labels
    batch_ids : np.ndarray, optional
        (N) Batch IDs
    per_cluster : bool, default True
        If `True`, computes the efficiency per truth cluster, than averages it

    Returns
    -------
    float
        Assignment efficiency
    """
    # If the vectors compared are empty, nothing to do
    if len(truth) == 0:
        return -1.0

    # Transform labels to be unique across all batch entries
    truth, _, truth_counts = unique_labels(truth, batch_ids)
    pred, _, pred_counts = unique_labels(pred, batch_ids)

    # Compute the contingency table
    table = contingency_table(truth, pred, len(truth_counts), len(pred_counts))

    # Evaluate the efficiency for each true cluster
    if per_cluster:
        efficiencies = table.max(axis=1) / truth_counts
        return efficiencies.mean()

    else:
        efficiency = np.sum(table.max(axis=1)) / len(truth)
        return efficiency


def pur_eff(truth, pred, batch_ids=None, per_cluster=True):
    """Assignment purity and efficiency.

    Parameters
    ----------
    truth : np.ndarray
        (N) Set of true labels
    pred : np.ndarray
        (N) Set of predicted labels
    batch_ids : np.ndarray, optional
        (N) Batch IDs
    per_cluster : bool, default True
        If `True`, computes the metrics per predicted cluster, than averages them

    Returns
    -------
    float
        Assignment purity
    float
        Assignment efficiency
    """
    # If the vectors compared are empty, nothing to do
    if len(truth) == 0:
        return -1.0, -1.0

    # Transform labels to be unique across all batch entries
    truth, _, truth_counts = unique_labels(truth, batch_ids)
    pred, _, pred_counts = unique_labels(pred, batch_ids)

    # Compute the contingency table
    table = contingency_table(truth, pred, len(truth_counts), len(pred_counts))

    # Evaluate the purity and efficiency
    if per_cluster:
        purities = table.max(axis=0) / pred_counts
        efficiencies = table.max(axis=1) / truth_counts

        return purities.mean(), efficiencies.mean()

    else:
        purity = np.sum(table.max(axis=0)) / len(pred)
        efficiency = np.sum(table.max(axis=1)) / len(truth)

        return purity, efficiency


def ari(truth, pred, batch_ids=None):
    """Computes the Adjusted Rand Index (ARI) between two sets of labels.

    Parameters
    ----------
    truth : np.ndarray
        (N) Set of true labels
    pred : np.ndarray
        (N) Set of predicted labels
    batch_ids : np.ndarray, optional
        (N) Batch IDs

    Returns
    -------
    float
        Adjusted Rand Index (ARI) value
    """
    # If the vectors compared are empty, nothing to do
    if len(truth) == 0:
        return -1.0

    # If required, transform labels to be unique across all batch entries
    if batch_ids is not None:
        truth = unique_labels(truth, batch_ids)[0]
        pred = unique_labels(pred, batch_ids)[0]

    return adjusted_rand_score(truth, pred)


def ami(truth, pred, batch_ids=None):
    """Computes the Adjusted Mutual Information (AMI) between two sets of labels.

    Parameters
    ----------
    truth : np.ndarray
        (N) Set of true labels
    pred : np.ndarray
        (N) Set of predicted labels
    batch_ids : np.ndarray, optional
        (N) Batch IDs

    Returns
    -------
    float
        Adjusted Mutual Information (AMI) value
    """
    # If the vectors compared are empty, nothing to do
    if len(truth) == 0:
        return -1.0

    # If required, transform labels to be unique across all batch entries
    if batch_ids is not None:
        truth = unique_labels(truth, batch_ids)[0]
        pred = unique_labels(pred, batch_ids)[0]

    return adjusted_mutual_info_score(truth, pred)


def sbd(truth, pred, batch_ids=None):
    """Compute the Symmetric Best Dice (SBD) score between two sets of labels.

    Parameters
    ----------
    truth : np.ndarray
        (N) Set of true labels
    pred : np.ndarray
        (N) Set of predicted labels
    batch_ids : np.ndarray, optional
        (N) Batch IDs

    Returns
    -------
    float
        Symmetric best dice value
    """
    # Transform labels to be unique across all batch entries
    truth, truth_unique, truth_counts = unique_labels(truth, batch_ids)
    pred, pred_unique, pred_counts = unique_labels(pred, batch_ids)

    # Compute the best dice both ways, take the minimum as the symmetric score
    bd1 = bd(truth, truth_unique, truth_counts, pred, pred_unique, pred_counts)
    bd2 = bd(pred, pred_unique, pred_counts, truth, truth_unique, truth_counts)

    return min(bd1, bd2)


def bd(truth, truth_unique, truth_counts, pred, pred_unique, pred_counts):
    """Computes the Best Dice (BD) between two sets of labels.

    Parameters
    ----------
    truth : np.ndarray
        (N) Set of true labels
    truth_unique : np.ndarray
        (K) Set of unique true labels
    truth_counts : np.ndarray
        (K) Number of realization of each unique true label
    pred : np.ndarray
        (N) Set of predicted labels
    pred_unique : np.ndarray
        (L) Set of unique predicted labels
    pred_counts : np.ndarray
        (L) Number of realization of each unique predicted label
    """
    # If the vectors compared are empty, nothing to do
    if len(truth) == 0:
        return -1.0

    # Loop over the predicted clusters
    total_bd = 0.0
    for i, c in enumerate(pred_unique):
        # Get the composition of the predicted cluster in the label array
        unique, counts = np.unique(truth[pred == c], return_counts=True)

        # Compute the best dice for this cluster
        best_dice = 0.0
        for j, d in enumerate(unique):
            dice = 2 * counts[j] / (pred_counts[i] + truth_counts[d])
            if dice > best_dice:
                best_dice = dice

        # Increment
        total_bd += best_dice

    # Take the mean best dice as a clustering score
    return total_bd / len(pred_unique)


def unique_labels(labels, batch_ids=None):
    """Transforms labels to range from 0 to C-1 labels (with C the number of
    unique values in the label array.

    If batch IDs are provided, ensures that the labels are unique at the batch
    level as well.

    Parameters
    ----------
    labels : np.ndarray
        (N) Labels
    batch_ids : np.ndarray, optional
        (N) Batch IDs

    Returns
    -------
    inverse : np.ndarray
        (N) Unique labels across all entries in the batch
    unique : np.ndarray
        (C) Unique set of labels
    counts : np.ndarray
        (C) Number of labels which belong to each unique category
    """
    if batch_ids is not None:
        labels = np.stack((labels, batch_ids))
    unique, inverse, counts = np.unique(
        labels, axis=-1, return_inverse=True, return_counts=True
    )

    return inverse, unique, counts
