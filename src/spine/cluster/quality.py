"""Cluster-to-truth overlap quality measurements.

The routines in this module assign each predicted cluster to its majority
truth instance and report the quality of that single assignment. They provide
the common geometric input used by model losses to reject ambiguous targets.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import numpy as np
from numba.typed import List as NumbaList  # pylint: disable=no-name-in-module

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.math.match import overlap_metrics

from .formation import form_clusters

__all__ = ["ClusterOverlapBatch", "get_cluster_overlap_batch"]


class ClusterOverlapBatch(NamedTuple):
    """Best truth-instance overlap measurements for predicted clusters.

    Attributes
    ----------
    match_ids : TensorBatch
        Truth-instance identifier matched to each predicted cluster.
    intersections : TensorBatch
        Number of shared voxels with the matched truth instance.
    purities : TensorBatch
        Fraction of each predicted cluster owned by its truth match.
    efficiencies : TensorBatch
        Fraction of each truth match recovered by the predicted cluster.
    ious : TensorBatch
        Intersection over union with the matched truth instance.
    """

    match_ids: TensorBatch
    intersections: TensorBatch
    purities: TensorBatch
    efficiencies: TensorBatch
    ious: TensorBatch


class _ClusterOverlap(NamedTuple):
    """Unbatched overlap arrays used internally by this module.

    The five arrays have one entry per predicted cluster and preserve predicted
    cluster order throughout event-local and batched calculations.
    """

    match_ids: np.ndarray
    intersections: np.ndarray
    purities: np.ndarray
    efficiencies: np.ndarray
    ious: np.ndarray


def get_cluster_overlap_batch(
    data: ClusterLabelBatch,
    clusts: IndexBatch,
    column: str = "group",
) -> ClusterOverlapBatch:
    """Measure each predicted cluster against its majority truth instance.

    The best truth match is the instance sharing the largest number of voxels
    with a predicted cluster. Purity, efficiency and IoU are all read from
    that same match, preserving a single and consistent instance assignment.
    Matches are performed independently in each batch entry, so truth IDs may
    safely repeat between events.

    Parameters
    ----------
    data : ClusterLabelBatch
        Structured voxel labels aligned with the cluster index space.
    clusts : IndexBatch
        (C) Predicted cluster indexes grouped by batch entry.
    column : str, default 'group'
        Integer-valued field used to form truth instances. Negative IDs are
        ignored and therefore cannot be selected as matches.

    Returns
    -------
    ClusterOverlapBatch
        Best match and overlap quality for every predicted cluster. Unmatched
        clusters receive match ID ``-1`` and zero-valued measurements.

    Raises
    ------
    ValueError
        If cluster and label batch spans are not aligned.
    """
    truth_ids = data.voxel_field(column).to_numpy()
    clusts_np = clusts.to_numpy()
    if not np.array_equal(clusts_np.spans, truth_ids.counts):
        raise ValueError("Cluster indexes and labels must share batch spans.")

    # Allocate flat outputs in predicted-cluster order
    num_clusts = len(clusts_np.index_list)
    output = _empty_cluster_overlap(num_clusts)
    truth_id_array = truth_ids.numpy_tensor().astype(np.int64, copy=False)

    # Match within each event so repeated truth identifiers remain independent.
    for batch_id in range(truth_ids.batch_size):
        lower, upper = clusts_np.edges[batch_id : batch_id + 2]
        if lower == upper:
            continue

        event_overlap = _get_batch_entry_overlap(
            truth_ids,
            clusts_np,
            truth_id_array,
            batch_id,
        )

        # Copy the event-local results into the flat batch arrays
        for output_values, event_values in zip(output, event_overlap, strict=True):
            output_values[lower:upper] = event_values

    counts = clusts_np.counts
    return ClusterOverlapBatch(
        TensorBatch(output.match_ids, counts),
        TensorBatch(output.intersections, counts),
        TensorBatch(output.purities, counts),
        TensorBatch(output.efficiencies, counts),
        TensorBatch(output.ious, counts),
    )


def _empty_cluster_overlap(num_clusts: int) -> _ClusterOverlap:
    """Initialize unmatched overlap results for predicted clusters.

    Parameters
    ----------
    num_clusts : int
        Number of predicted clusters represented by the output.

    Returns
    -------
    _ClusterOverlap
        Match IDs initialized to ``-1`` and zero-valued overlap metrics.
    """
    return _ClusterOverlap(
        np.full(num_clusts, -1, dtype=np.int64),
        np.zeros(num_clusts, dtype=np.int64),
        np.zeros(num_clusts, dtype=np.float32),
        np.zeros(num_clusts, dtype=np.float32),
        np.zeros(num_clusts, dtype=np.float32),
    )


def _get_batch_entry_overlap(
    truth_id_batch: TensorBatch,
    clusts: IndexBatch,
    truth_ids: np.ndarray,
    batch_id: int,
) -> _ClusterOverlap:
    """Measure one entry from aligned label and cluster batches.

    Global voxel indexes are translated to event-local indexes before invoking
    the unbatched overlap kernel.
    """
    lower, upper = clusts.edges[batch_id : batch_id + 2]
    voxel_lower, voxel_upper = truth_id_batch.edges[batch_id : batch_id + 2]
    truth_ids_b = truth_ids[voxel_lower:voxel_upper]
    predicted = [
        np.asarray(index - voxel_lower, dtype=np.int64)
        for index in clusts.index_list[lower:upper]
    ]
    return _get_cluster_overlap(truth_ids_b, predicted)


def _get_cluster_overlap(
    truth_ids: np.ndarray,
    predicted: list[np.ndarray],
) -> _ClusterOverlap:
    """Measure one event's predicted clusters against its truth instances.

    Parameters
    ----------
    truth_ids : np.ndarray
        Truth-instance ID for every voxel in the event.
    predicted : list[np.ndarray]
        Sorted event-local voxel indexes for each predicted cluster.

    Returns
    -------
    _ClusterOverlap
        Majority match and corresponding qualities for each prediction.
    """
    output = _empty_cluster_overlap(len(predicted))
    unique_truth = np.unique(truth_ids[truth_ids >= 0])
    if len(unique_truth) == 0:
        return output

    # Group the truth indexes with one stable sort. Predicted indexes are
    # already sorted by their builders, as required by the linear kernel.
    truth, _ = form_clusters(truth_ids)
    counts, purities, efficiencies, ious = overlap_metrics(
        cast(Any, NumbaList)(predicted),
        cast(Any, NumbaList)(truth),
    )

    # Select the majority truth instance once, then gather every metric at that
    # same matrix entry to preserve a coherent object-to-truth assignment.
    best = np.argmax(counts, axis=1)
    rows = np.arange(len(predicted))
    best_counts = counts[rows, best]
    matched = best_counts > 0
    matched_rows = rows[matched]
    matched_truth = best[matched]
    output.match_ids[matched_rows] = unique_truth[matched_truth]
    output.intersections[matched_rows] = best_counts[matched]
    output.purities[matched_rows] = purities[matched_rows, matched_truth]
    output.efficiencies[matched_rows] = efficiencies[matched_rows, matched_truth]
    output.ious[matched_rows] = ious[matched_rows, matched_truth]
    return output
