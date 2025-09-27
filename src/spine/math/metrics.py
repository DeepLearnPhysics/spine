"""Numba JIT compiled implementation of clustering evaluation metrics.

This module provides efficient implementations of:
- Adjusted Rand Index (ARI)
- Adjusted Mutual Information (AMI)
"""

import numba as nb
import numpy as np

from .linalg import contingency_table

__all__ = ["adjusted_rand_score", "adjusted_mutual_info_score"]


@nb.jit(nopython=True, cache=True)
def _comb2(n):
    """Compute binomial coefficient n choose 2.

    Parameters
    ----------
    n : int
        Number of items to choose from.

    Returns
    -------
    int
        The number of ways to choose 2 items from n items, equal to n*(n-1)/2.
    """
    return n * (n - 1) // 2


@nb.jit(nopython=True, cache=True)
def adjusted_rand_score(labels_true, labels_pred):
    """Compute the Adjusted Rand Index (ARI) between two clusterings.

    The Adjusted Rand Index is a measure of the similarity between two
    data clusterings. It is a function that measures the similarity of the two
    assignments, ignoring permutations and correcting for chance agreement.

    The ARI is bounded between -1 and 1:
    - 1.0 indicates perfect clustering agreement
    - 0.0 indicates random clustering (expected value for independent labelings)
    - Negative values indicate worse than random clustering

    The formula is: ARI = (RI - E[RI]) / (max(RI) - E[RI])
    where RI is the Rand Index and E[RI] is the expected Rand Index under
    random labelings.

    Parameters
    ----------
    labels_true : ndarray of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : ndarray of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    ari : float
        Adjusted Rand Index. A clustering result satisfying the constraints
        of a correct clustering has a score of 1.0.

    Notes
    -----
    This implementation uses a fast numba-compiled algorithm that avoids
    constructing the full pairwise similarity matrix.

    References
    ----------
    .. [1] Hubert, L. and Arabie, P. (1985). "Comparing partitions."
           Journal of Classification 2(1): 193-218.
    .. [2] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

    Examples
    --------
    Perfect clustering:
    >>> labels_true = [0, 0, 1, 1]
    >>> labels_pred = [0, 0, 1, 1]
    >>> adjusted_rand_score(labels_true, labels_pred)
    1.0

    Random clustering:
    >>> labels_true = [0, 0, 1, 1]
    >>> labels_pred = [0, 1, 0, 1]
    >>> adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
    0.0
    """
    # Get dimensions for contingency table
    nx = labels_true.max() + 1 if len(labels_true) > 0 else 1
    ny = labels_pred.max() + 1 if len(labels_pred) > 0 else 1
    contingency = contingency_table(labels_true, labels_pred, nx, ny)

    # Compute sums
    sum_comb_c = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            sum_comb_c += _comb2(contingency[i, j])

    sum_comb_k = 0  # Sum over rows (true clusters)
    for i in range(contingency.shape[0]):
        row_sum = 0
        for j in range(contingency.shape[1]):
            row_sum += contingency[i, j]
        sum_comb_k += _comb2(row_sum)

    sum_comb_c_pred = 0  # Sum over columns (pred clusters)
    for j in range(contingency.shape[1]):
        col_sum = 0
        for i in range(contingency.shape[0]):
            col_sum += contingency[i, j]
        sum_comb_c_pred += _comb2(col_sum)

    n_samples = len(labels_true)
    sum_comb_n = _comb2(n_samples)

    expected_index = sum_comb_k * sum_comb_c_pred / sum_comb_n
    max_index = (sum_comb_k + sum_comb_c_pred) / 2.0

    if max_index == expected_index:
        return 1.0

    return (sum_comb_c - expected_index) / (max_index - expected_index)


@nb.jit(nopython=True, cache=True)
def _entropy(labels):
    """Compute entropy of a labeling."""
    unique_labels = np.unique(labels)
    n = len(labels)

    if n <= 1:
        return 0.0

    entropy = 0.0
    for label in unique_labels:
        count = np.sum(labels == label)
        if count > 0:
            p = count / n
            entropy -= p * np.log(p)

    return entropy


@nb.jit(nopython=True, cache=True)
def _mutual_info(labels_true, labels_pred):
    """Compute mutual information between two clusterings."""
    # Get dimensions for contingency table
    nx = labels_true.max() + 1 if len(labels_true) > 0 else 1
    ny = labels_pred.max() + 1 if len(labels_pred) > 0 else 1
    contingency = contingency_table(labels_true, labels_pred, nx, ny)

    n_samples = len(labels_true)
    mi = 0.0

    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            n_ij = contingency[i, j]
            if n_ij == 0:
                continue

            # Marginal counts
            n_i = 0
            for k in range(contingency.shape[1]):
                n_i += contingency[i, k]

            n_j = 0
            for k in range(contingency.shape[0]):
                n_j += contingency[k, j]

            mi += n_ij / n_samples * np.log((n_samples * n_ij) / (n_i * n_j))

    return mi


@nb.jit(nopython=True, cache=True)
def adjusted_mutual_info_score(labels_true, labels_pred):
    """Compute the Adjusted Mutual Information (AMI) between two clusterings.

    The Adjusted Mutual Information is a measure of agreement between two
    partitions, adjusted for chance. It employs the expected mutual information
    under a hypergeometric model of randomness.

    The AMI is normalized between 0 and 1:
    - 1.0 indicates perfect clustering agreement
    - 0.0 indicates independent labelings (expected value for random labelings)
    - Values close to 0.0 indicate near-random agreement

    The formula is: AMI = (MI - E[MI]) / (max(H(U), H(V)) - E[MI])
    where MI is the mutual information, E[MI] is the expected mutual
    information, and H(U), H(V) are the entropies of the two labelings.

    Parameters
    ----------
    labels_true : ndarray of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : ndarray of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    ami : float
        Adjusted Mutual Information score. Perfect labelings are scored 1.0.
        Bad labelings or independent labelings have non-positive scores.

    Notes
    -----
    This implementation uses a fast numba-compiled algorithm that computes
    the hypergeometric expected mutual information directly from the
    contingency table.

    References
    ----------
    .. [1] Vinh, N. X., Epps, J., & Bailey, J. (2010). "Information theoretic
           measures for clusterings comparison: Variants, properties,
           normalization and correction for chance." Journal of Machine
           Learning Research, 11, 2837-2854.
    .. [2] https://en.wikipedia.org/wiki/Adjusted_mutual_information

    Examples
    --------
    Perfect clustering:
    >>> labels_true = [0, 0, 1, 1]
    >>> labels_pred = [0, 0, 1, 1]
    >>> adjusted_mutual_info_score(labels_true, labels_pred)
    1.0

    Random clustering:
    >>> labels_true = [0, 0, 1, 1]
    >>> labels_pred = [0, 1, 0, 1]
    >>> adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
    0.0...
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("Labels must have the same length")

    if len(labels_true) == 0:
        return 1.0

    # Handle trivial cases
    n_true_clusters = len(np.unique(labels_true))
    n_pred_clusters = len(np.unique(labels_pred))

    if n_true_clusters == 1 and n_pred_clusters == 1:
        return 1.0
    elif n_true_clusters == 1 or n_pred_clusters == 1:
        return 0.0

    # Compute entropies and mutual information
    entropy_true = _entropy(labels_true)
    entropy_pred = _entropy(labels_pred)
    mutual_info = _mutual_info(labels_true, labels_pred)

    # Expected mutual information (approximation for large n)
    n_samples = len(labels_true)
    expected_mi = entropy_true * entropy_pred / np.log(n_samples)

    # Compute adjusted mutual information
    mean_entropy = (entropy_true + entropy_pred) / 2.0

    if mean_entropy == expected_mi:
        return 1.0

    return (mutual_info - expected_mi) / (mean_entropy - expected_mi)
