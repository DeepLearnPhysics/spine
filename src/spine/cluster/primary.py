"""Shared shower-primary targets for objectives and analysis."""

from __future__ import annotations

import numpy as np

from spine.cluster.label import (
    get_cluster_closest_primary_label_batch,
    get_cluster_label_batch,
)
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch


def primary_purity_mask_batch(
    group_ids: TensorBatch, primary_ids: TensorBatch
) -> np.ndarray:
    """Select nodes in event-local groups with exactly one primary target.

    Parameters
    ----------
    group_ids : TensorBatch
        Group assignments for each node, partitioned by event.
    primary_ids : TensorBatch
        Binary primary targets with the same node ordering and event counts.

    Returns
    -------
    numpy.ndarray
        Boolean node eligibility mask. Group IDs are local to each event.
    """
    groups = group_ids.to_numpy()
    targets = primary_ids.to_numpy()
    valid = np.zeros(len(targets.data), dtype=bool)
    for event in range(groups.batch_size):
        lower, upper = groups.edges[event : event + 2]
        group = groups.data[lower:upper]
        target = targets.data[lower:upper]
        for group_id in np.unique(group):
            mask = group == group_id
            if np.count_nonzero(target[mask] == 1) == 1:
                valid[lower:upper] |= mask
    return valid


def shower_primary_target(
    labels: ClusterLabelBatch,
    clusters: IndexBatch,
    coordinates: TensorBatch | None = None,
    *,
    use_closest: bool = False,
    high_purity: bool = False,
    groups: TensorBatch | None = None,
) -> tuple[TensorBatch, np.ndarray]:
    """Build the same node targets for training and reporting.

    Closest-fragment selection only relabels fragments of the true primary
    particle, within each shower group. With ``high_purity``, an explicit
    ``groups`` batch selects predicted groups; otherwise truth groups are used.
    Overlap-quality requirements are applied by the caller.

    Parameters
    ----------
    labels : ClusterLabelBatch
        Structured truth on the reconstructed point domain.
    clusters : IndexBatch
        Reconstructed shower-fragment indexes on that domain.
    coordinates : TensorBatch, optional
        True particle start coordinates required for closest selection.
    use_closest : bool, default False
        Resolve split primary fragments using their distance to the start.
    high_purity : bool, default False
        Accept only groups containing exactly one primary target.
    groups : TensorBatch, optional
        Predicted grouping, aligned with ``clusters``.

    Returns
    -------
    TensorBatch
        Binary targets, including invalid truth labels where applicable.
    numpy.ndarray
        Eligibility mask aligned with the targets.
    """
    targets = get_cluster_label_batch(labels, clusters, "group_primary")
    valid = targets.to_numpy().data > -1
    if use_closest:
        if coordinates is None:
            raise ValueError("Closest shower-primary targets require `coord_label`.")
        targets = get_cluster_closest_primary_label_batch(
            labels, coordinates, clusters, targets
        )
        valid &= targets.to_numpy().data > -1
    if high_purity:
        if groups is None:
            groups = get_cluster_label_batch(labels, clusters, "group")
        valid &= primary_purity_mask_batch(groups, targets)
    return targets, valid
