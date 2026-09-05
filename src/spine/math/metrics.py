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
    "cluster_metrics",
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
    return cluster_metrics(labels_true, labels_pred, "ari")["ari"]


@nb.jit(nopython=True, cache=True)
def _adjusted_rand_from_table(
    contingency,
    truth_counts,
    pred_counts,
    num_samples,
):
    """Compute ARI from a precomputed contingency table.

    Parameters
    ----------
    contingency : np.ndarray
        ``(C_t, C_p)`` truth-versus-prediction assignment counts.
    truth_counts : np.ndarray
        ``(C_t)`` truth cluster populations.
    pred_counts : np.ndarray
        ``(C_p)`` predicted cluster populations.
    num_samples : int
        Number of comparable assignments represented by the table.

    Returns
    -------
    float
        Adjusted Rand Index, or ``NaN`` below two samples.
    """
    if num_samples < 2:
        return np.nan

    # Count pairs shared by each truth/prediction cluster intersection.
    sum_comb_c = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            sum_comb_c += _comb2(contingency[i, j])

    sum_comb_k = 0
    for count in truth_counts:
        sum_comb_k += _comb2(count)

    sum_comb_c_pred = 0
    for count in pred_counts:
        sum_comb_c_pred += _comb2(count)
    sum_comb_n = _comb2(num_samples)
    expected_index = sum_comb_k * sum_comb_c_pred / sum_comb_n
    max_index = (sum_comb_k + sum_comb_c_pred) / 2.0

    # Identical degenerate partitions have no nonzero adjustment denominator.
    if max_index == expected_index:
        return 1.0

    return (sum_comb_c - expected_index) / (max_index - expected_index)


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
    return cluster_metrics(labels_true, labels_pred, "ami")["ami"]


@nb.jit(nopython=True, cache=True)
def _adjusted_mutual_info_from_table(
    contingency,
    truth_counts,
    pred_counts,
    num_samples,
):
    """Compute AMI from a precomputed contingency table.

    Parameters
    ----------
    contingency : np.ndarray
        ``(C_t, C_p)`` truth-versus-prediction assignment counts.
    truth_counts : np.ndarray
        ``(C_t)`` truth cluster populations.
    pred_counts : np.ndarray
        ``(C_p)`` predicted cluster populations.
    num_samples : int
        Number of comparable assignments represented by the table.

    Returns
    -------
    float
        Adjusted Mutual Information for a nonempty table.
    """
    if len(truth_counts) == 1 and len(pred_counts) == 1:
        return 1.0
    if len(truth_counts) == 1 or len(pred_counts) == 1:
        return 0.0

    entropy_true = 0.0
    for count in truth_counts:
        probability = count / num_samples
        entropy_true -= probability * np.log(probability)

    entropy_pred = 0.0
    for count in pred_counts:
        probability = count / num_samples
        entropy_pred -= probability * np.log(probability)

    mutual_info = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            count = contingency[i, j]
            if count > 0:
                mutual_info += (
                    count
                    / num_samples
                    * np.log((num_samples * count) / (truth_counts[i] * pred_counts[j]))
                )

    expected_mi = entropy_true * entropy_pred / np.log(num_samples)
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


def _normalize_metric_names(metric_names):
    """Normalize and validate a requested clustering metric sequence."""
    if isinstance(metric_names, str):
        metric_names = (metric_names,)

    metric_names = tuple(dict.fromkeys(metric_names))
    supported = {"pur", "eff", "ari", "ami", "sbd"}
    invalid = set(metric_names) - supported
    if invalid:
        raise ValueError(f"Unsupported clustering metrics: {sorted(invalid)}")

    return metric_names


def _purity_from_table(table, pred_counts, num_samples, per_cluster):
    """Compute purity from a contingency table and prediction counts."""
    dominant_truth = table.max(axis=0)
    if per_cluster:
        return float(np.mean(dominant_truth / pred_counts))

    return float(np.sum(dominant_truth) / num_samples)


def _efficiency_from_table(table, truth_counts, num_samples, per_cluster):
    """Compute efficiency from a contingency table and truth counts."""
    dominant_pred = table.max(axis=1)
    if per_cluster:
        return float(np.mean(dominant_pred / truth_counts))

    return float(np.sum(dominant_pred) / num_samples)


def _sbd_from_table(table, truth_counts, pred_counts):
    """Compute symmetric Best Dice from a contingency table."""
    denominators = truth_counts[:, None] + pred_counts[None, :]
    dice = 2.0 * table / denominators
    truth_to_pred = np.mean(dice.max(axis=0))
    pred_to_truth = np.mean(dice.max(axis=1))
    return float(min(truth_to_pred, pred_to_truth))


def cluster_metrics(
    truth,
    pred,
    metric_names=("pur", "eff", "ari"),
    batch_ids=None,
    per_cluster=True,
):
    """Compute multiple clustering metrics from shared sufficient statistics.

    Both assignments are densified once and used to build a single contingency
    table. Purity, efficiency, ARI, AMI, and symmetric Best Dice are then
    derived from that table without repeating the dominant preprocessing work.

    Parameters
    ----------
    truth : np.ndarray
        ``(N)`` truth cluster IDs.
    pred : np.ndarray
        ``(N)`` predicted cluster IDs.
    metric_names : str or sequence of str, default ('pur', 'eff', 'ari')
        Metrics to compute. Supported values are ``pur``, ``eff``, ``ari``,
        ``ami`` and ``sbd``.
    batch_ids : np.ndarray, optional
        ``(N)`` event IDs used to keep local cluster IDs event-specific.
    per_cluster : bool, default True
        Average purity and efficiency equally over their defining clusters. If
        ``False``, weight those clusters by their number of assignments.

    Returns
    -------
    dict[str, float]
        Requested metric values keyed by name. Every value is ``NaN`` when no
        comparable assignments are provided; ARI is also ``NaN`` for a single
        assignment.

    Raises
    ------
    ValueError
        If the assignments have different lengths, batch IDs are misaligned,
        or an unsupported metric is requested.
    """
    metric_names = _normalize_metric_names(metric_names)

    if len(truth) != len(pred):
        raise ValueError("Labels must have the same length")
    if batch_ids is not None and len(batch_ids) != len(truth):
        raise ValueError("Batch IDs must have the same length as labels")
    if len(truth) == 0 or len(metric_names) == 0:
        return {name: np.nan for name in metric_names}

    # Densification makes the contingency table compact even when source IDs
    # are sparse, large, or repeated independently in different events.
    truth, _, truth_counts = unique_labels(truth, batch_ids)
    pred, _, pred_counts = unique_labels(pred, batch_ids)
    table = contingency_table(
        truth,
        pred,
        len(truth_counts),
        len(pred_counts),
    )
    num_samples = len(truth)

    results = {}
    if "pur" in metric_names:
        results["pur"] = _purity_from_table(
            table, pred_counts, num_samples, per_cluster
        )

    if "eff" in metric_names:
        results["eff"] = _efficiency_from_table(
            table, truth_counts, num_samples, per_cluster
        )

    if "ari" in metric_names:
        results["ari"] = float(
            _adjusted_rand_from_table(
                table,
                truth_counts,
                pred_counts,
                num_samples,
            )
        )

    if "ami" in metric_names:
        results["ami"] = float(
            _adjusted_mutual_info_from_table(
                table,
                truth_counts,
                pred_counts,
                num_samples,
            )
        )

    if "sbd" in metric_names:
        results["sbd"] = _sbd_from_table(table, truth_counts, pred_counts)

    # Preserve the requested ordering rather than the implementation order.
    return {name: results[name] for name in metric_names}


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
    return cluster_metrics(
        truth,
        pred,
        "pur",
        batch_ids,
        per_cluster,
    )["pur"]


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
    return cluster_metrics(
        truth,
        pred,
        "eff",
        batch_ids,
        per_cluster,
    )["eff"]


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
    results = cluster_metrics(
        truth,
        pred,
        ("pur", "eff"),
        batch_ids,
        per_cluster,
    )
    return results["pur"], results["eff"]


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
    return cluster_metrics(truth, pred, "ari", batch_ids)["ari"]


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
    return cluster_metrics(truth, pred, "ami", batch_ids)["ami"]


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
    return cluster_metrics(truth, pred, "sbd", batch_ids)["sbd"]


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
