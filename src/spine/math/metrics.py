"""Clustering-assignment metrics and their batch-aware wrappers.

This module provides Numba-accelerated Adjusted Rand Index (ARI) and
Adjusted Mutual Information (AMI) implementations alongside purity,
efficiency, and best-Dice scores. All public metrics use ``NaN`` when no
comparable assignments exist; finite values, including negative ARI values,
always represent genuine metric results.
"""

import numba as nb
import numpy as np

from spine.math.linalg import contingency_table

__all__ = [
    "ami",
    "ari",
    "bd",
    "eff",
    "pur",
    "pur_eff",
    "sbd",
    "unique_labels",
]


@nb.jit(nopython=True, cache=True)
def _comb2(n):
    """Return the number of unordered pairs among ``n`` items.

    Parameters
    ----------
    n : int
        Number of items.

    Returns
    -------
    int
        Binomial coefficient ``n choose 2``.
    """
    return n * (n - 1) // 2


@nb.jit(nopython=True, cache=True)
def _adjusted_rand_score(labels_true, labels_pred):
    """Compute the Adjusted Rand Index between two cluster assignments.

    ARI compares whether every pair of samples is grouped consistently in the
    two partitions, then corrects the agreement expected by chance. It is
    invariant under a permutation of cluster IDs and may legitimately be
    negative when agreement is worse than random.

    Parameters
    ----------
    labels_true : np.ndarray
        ``(N)`` reference cluster IDs.
    labels_pred : np.ndarray
        ``(N)`` predicted cluster IDs.

    Returns
    -------
    float
        Adjusted Rand Index. Returns ``NaN`` when fewer than two assignments
        are provided, ``1`` for identical nonempty single-cluster partitions,
        and the standard ARI otherwise.

    Raises
    ------
    ValueError
        If the assignment arrays do not have the same length.

    Notes
    -----
    The implementation follows the chance-corrected formulation of Hubert and
    Arabie and avoids constructing a pairwise ``N x N`` matrix.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("Labels must have the same length")

    # ARI is pair-based and therefore undefined below two comparable samples.
    if len(labels_true) < 2:
        return np.nan

    nx = labels_true.max() + 1
    ny = labels_pred.max() + 1
    contingency = contingency_table(labels_true, labels_pred, nx, ny)

    # Count pairs shared by each truth/prediction cluster intersection.
    sum_comb_c = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            sum_comb_c += _comb2(contingency[i, j])

    sum_comb_k = 0
    for i in range(contingency.shape[0]):
        row_sum = 0
        for j in range(contingency.shape[1]):
            row_sum += contingency[i, j]
        sum_comb_k += _comb2(row_sum)

    sum_comb_c_pred = 0
    for j in range(contingency.shape[1]):
        col_sum = 0
        for i in range(contingency.shape[0]):
            col_sum += contingency[i, j]
        sum_comb_c_pred += _comb2(col_sum)

    sum_comb_n = _comb2(len(labels_true))
    expected_index = sum_comb_k * sum_comb_c_pred / sum_comb_n
    max_index = (sum_comb_k + sum_comb_c_pred) / 2.0

    # Identical degenerate partitions have no nonzero adjustment denominator.
    if max_index == expected_index:
        return 1.0

    return (sum_comb_c - expected_index) / (max_index - expected_index)


@nb.jit(nopython=True, cache=True)
def _entropy(labels):
    """Compute the Shannon entropy of a cluster assignment.

    Parameters
    ----------
    labels : np.ndarray
        ``(N)`` cluster IDs.

    Returns
    -------
    float
        Natural-log entropy of the empirical cluster distribution. Empty and
        singleton assignments have zero entropy.
    """
    unique = np.unique(labels)
    num_labels = len(labels)

    if num_labels <= 1:
        return 0.0

    entropy = 0.0
    for label in unique:
        count = np.sum(labels == label)
        if count > 0:
            probability = count / num_labels
            entropy -= probability * np.log(probability)

    return entropy


@nb.jit(nopython=True, cache=True)
def _mutual_info(labels_true, labels_pred):
    """Compute mutual information between two cluster assignments.

    Parameters
    ----------
    labels_true : np.ndarray
        ``(N)`` reference cluster IDs.
    labels_pred : np.ndarray
        ``(N)`` predicted cluster IDs.

    Returns
    -------
    float
        Mutual information derived from the assignment contingency table.
    """
    nx = labels_true.max() + 1
    ny = labels_pred.max() + 1
    contingency = contingency_table(labels_true, labels_pred, nx, ny)

    num_samples = len(labels_true)
    mutual_info = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            count = contingency[i, j]
            if count == 0:
                continue

            truth_count = 0
            for k in range(contingency.shape[1]):
                truth_count += contingency[i, k]

            pred_count = 0
            for k in range(contingency.shape[0]):
                pred_count += contingency[k, j]

            mutual_info += (
                count
                / num_samples
                * np.log((num_samples * count) / (truth_count * pred_count))
            )

    return mutual_info


@nb.jit(nopython=True, cache=True)
def _adjusted_mutual_info_score(labels_true, labels_pred):
    """Compute Adjusted Mutual Information between two cluster assignments.

    AMI measures shared information between partitions and corrects it for
    chance agreement. Cluster ID values themselves are immaterial.

    Parameters
    ----------
    labels_true : np.ndarray
        ``(N)`` reference cluster IDs.
    labels_pred : np.ndarray
        ``(N)`` predicted cluster IDs.

    Returns
    -------
    float
        Adjusted Mutual Information. Returns ``NaN`` for empty assignments,
        ``1`` when both partitions contain one nonempty cluster, and ``0``
        when exactly one of the partitions contains one cluster.

    Raises
    ------
    ValueError
        If the assignment arrays do not have the same length.

    Notes
    -----
    Expected mutual information is approximated from the partition entropies,
    which keeps this implementation inexpensive for large assignments.

    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("Labels must have the same length")

    if len(labels_true) == 0:
        return np.nan

    num_truth_clusters = len(np.unique(labels_true))
    num_pred_clusters = len(np.unique(labels_pred))
    if num_truth_clusters == 1 and num_pred_clusters == 1:
        return 1.0
    if num_truth_clusters == 1 or num_pred_clusters == 1:
        return 0.0

    entropy_true = _entropy(labels_true)
    entropy_pred = _entropy(labels_pred)
    mutual_info = _mutual_info(labels_true, labels_pred)

    expected_mi = entropy_true * entropy_pred / np.log(len(labels_true))
    mean_entropy = (entropy_true + entropy_pred) / 2.0
    if mean_entropy == expected_mi:
        return 1.0

    return (mutual_info - expected_mi) / (mean_entropy - expected_mi)


def unique_labels(labels, batch_ids=None):
    """Convert arbitrary cluster IDs into contiguous assignment IDs.

    When batch IDs are supplied, each ``(batch, cluster)`` pair is treated as
    a distinct cluster. This prevents equal local cluster IDs in different
    events from being merged during batch-level metric evaluation.

    Parameters
    ----------
    labels : np.ndarray
        ``(N)`` cluster IDs.
    batch_ids : np.ndarray, optional
        ``(N)`` event IDs aligned with ``labels``.

    Returns
    -------
    inverse : np.ndarray
        ``(N)`` contiguous cluster IDs in ``[0, C)``.
    unique : np.ndarray
        Original unique IDs, or unique ``(label, batch)`` pairs when batching.
    counts : np.ndarray
        ``(C)`` number of assignments in each contiguous cluster.
    """
    if batch_ids is not None:
        labels = np.stack((labels, batch_ids))
    unique, inverse, counts = np.unique(
        labels, axis=-1, return_inverse=True, return_counts=True
    )

    return inverse, unique, counts


def pur(truth, pred, batch_ids=None, per_cluster=True):
    """Compute clustering purity.

    Purity measures the fraction of each predicted cluster belonging to its
    dominant truth cluster.

    Parameters
    ----------
    truth : np.ndarray
        ``(N)`` truth cluster IDs.
    pred : np.ndarray
        ``(N)`` predicted cluster IDs.
    batch_ids : np.ndarray, optional
        ``(N)`` event IDs used to keep local cluster IDs event-specific.
    per_cluster : bool, default True
        Average equally over predicted clusters. If ``False``, weight each
        predicted cluster by its number of assignments.

    Returns
    -------
    float
        Assignment purity in ``[0, 1]``, or ``NaN`` when the inputs are empty.
    """
    if len(truth) == 0:
        return np.nan

    truth, _, truth_counts = unique_labels(truth, batch_ids)
    pred, _, pred_counts = unique_labels(pred, batch_ids)
    table = contingency_table(truth, pred, len(truth_counts), len(pred_counts))

    if per_cluster:
        purities = table.max(axis=0) / pred_counts
        return purities.mean()

    return np.sum(table.max(axis=0)) / len(pred)


def eff(truth, pred, batch_ids=None, per_cluster=True):
    """Compute clustering efficiency.

    Efficiency measures the fraction of each truth cluster captured by its
    best matching predicted cluster.

    Parameters
    ----------
    truth : np.ndarray
        ``(N)`` truth cluster IDs.
    pred : np.ndarray
        ``(N)`` predicted cluster IDs.
    batch_ids : np.ndarray, optional
        ``(N)`` event IDs used to keep local cluster IDs event-specific.
    per_cluster : bool, default True
        Average equally over truth clusters. If ``False``, weight each truth
        cluster by its number of assignments.

    Returns
    -------
    float
        Assignment efficiency in ``[0, 1]``, or ``NaN`` when the inputs are
        empty.
    """
    if len(truth) == 0:
        return np.nan

    truth, _, truth_counts = unique_labels(truth, batch_ids)
    pred, _, pred_counts = unique_labels(pred, batch_ids)
    table = contingency_table(truth, pred, len(truth_counts), len(pred_counts))

    if per_cluster:
        efficiencies = table.max(axis=1) / truth_counts
        return efficiencies.mean()

    return np.sum(table.max(axis=1)) / len(truth)


def pur_eff(truth, pred, batch_ids=None, per_cluster=True):
    """Compute clustering purity and efficiency from one contingency table.

    Parameters
    ----------
    truth : np.ndarray
        ``(N)`` truth cluster IDs.
    pred : np.ndarray
        ``(N)`` predicted cluster IDs.
    batch_ids : np.ndarray, optional
        ``(N)`` event IDs used to keep local cluster IDs event-specific.
    per_cluster : bool, default True
        Average each score equally over the clusters that define it. If
        ``False``, weight clusters by their number of assignments.

    Returns
    -------
    purity : float
        Assignment purity in ``[0, 1]``.
    efficiency : float
        Assignment efficiency in ``[0, 1]``. Both results are ``NaN`` when
        the inputs are empty.
    """
    if len(truth) == 0:
        return np.nan, np.nan

    truth, _, truth_counts = unique_labels(truth, batch_ids)
    pred, _, pred_counts = unique_labels(pred, batch_ids)
    table = contingency_table(truth, pred, len(truth_counts), len(pred_counts))

    if per_cluster:
        purities = table.max(axis=0) / pred_counts
        efficiencies = table.max(axis=1) / truth_counts
        return purities.mean(), efficiencies.mean()

    purity = np.sum(table.max(axis=0)) / len(pred)
    efficiency = np.sum(table.max(axis=1)) / len(truth)
    return purity, efficiency


def ari(truth, pred, batch_ids=None):
    """Compute batch-aware Adjusted Rand Index.

    Parameters
    ----------
    truth : np.ndarray
        ``(N)`` truth cluster IDs.
    pred : np.ndarray
        ``(N)`` predicted cluster IDs.
    batch_ids : np.ndarray, optional
        ``(N)`` event IDs used to keep local cluster IDs event-specific.

    Returns
    -------
    float
        Adjusted Rand Index, or ``NaN`` when fewer than two assignments are
        comparable.
    """
    if len(truth) < 2:
        return np.nan

    if batch_ids is not None:
        truth = unique_labels(truth, batch_ids)[0]
        pred = unique_labels(pred, batch_ids)[0]

    return _adjusted_rand_score(truth, pred)


def ami(truth, pred, batch_ids=None):
    """Compute batch-aware Adjusted Mutual Information.

    Parameters
    ----------
    truth : np.ndarray
        ``(N)`` truth cluster IDs.
    pred : np.ndarray
        ``(N)`` predicted cluster IDs.
    batch_ids : np.ndarray, optional
        ``(N)`` event IDs used to keep local cluster IDs event-specific.

    Returns
    -------
    float
        Adjusted Mutual Information, or ``NaN`` when the inputs are empty.
    """
    if len(truth) == 0:
        return np.nan

    if batch_ids is not None:
        truth = unique_labels(truth, batch_ids)[0]
        pred = unique_labels(pred, batch_ids)[0]

    return _adjusted_mutual_info_score(truth, pred)


def sbd(truth, pred, batch_ids=None):
    """Compute Symmetric Best Dice between two cluster assignments.

    Best Dice is evaluated in both directions and the smaller score is kept,
    penalizing both truth fragmentation and reconstructed cluster merging.

    Parameters
    ----------
    truth : np.ndarray
        ``(N)`` truth cluster IDs.
    pred : np.ndarray
        ``(N)`` predicted cluster IDs.
    batch_ids : np.ndarray, optional
        ``(N)`` event IDs used to keep local cluster IDs event-specific.

    Returns
    -------
    float
        Symmetric Best Dice in ``[0, 1]``, or ``NaN`` when the inputs are
        empty.
    """
    if len(truth) == 0:
        return np.nan

    truth, _, truth_counts = unique_labels(truth, batch_ids)
    pred, _, pred_counts = unique_labels(pred, batch_ids)
    truth_unique = np.arange(len(truth_counts))
    pred_unique = np.arange(len(pred_counts))

    truth_to_pred = bd(
        truth,
        truth_unique,
        truth_counts,
        pred,
        pred_unique,
        pred_counts,
    )
    pred_to_truth = bd(
        pred,
        pred_unique,
        pred_counts,
        truth,
        truth_unique,
        truth_counts,
    )
    return min(truth_to_pred, pred_to_truth)


def bd(truth, truth_unique, truth_counts, pred, pred_unique, pred_counts):
    """Compute directional Best Dice from prediction to truth clusters.

    Parameters
    ----------
    truth : np.ndarray
        ``(N)`` contiguous truth cluster IDs.
    truth_unique : np.ndarray
        ``(K)`` unique truth cluster IDs. Retained for the public historical
        interface; ``truth_counts`` provides the lookup needed internally.
    truth_counts : np.ndarray
        ``(K)`` number of assignments in each truth cluster.
    pred : np.ndarray
        ``(N)`` contiguous predicted cluster IDs.
    pred_unique : np.ndarray
        ``(L)`` unique predicted cluster IDs.
    pred_counts : np.ndarray
        ``(L)`` number of assignments in each predicted cluster.

    Returns
    -------
    float
        Mean best Dice overlap over predicted clusters, or ``NaN`` when the
        assignments are empty.
    """
    if len(truth) == 0:
        return np.nan

    total_bd = 0.0
    for i, pred_id in enumerate(pred_unique):
        unique, counts = np.unique(truth[pred == pred_id], return_counts=True)

        best_dice = 0.0
        for j, truth_id in enumerate(unique):
            dice = 2 * counts[j] / (pred_counts[i] + truth_counts[truth_id])
            best_dice = max(best_dice, dice)

        total_bd += best_dice

    return total_bd / len(pred_unique)
