"""Shared truth-overlap quality requirements for model objectives.

This module separates the policy used to accept a reconstructed object from
the loss which consumes that object. It also provides a small, forward-local
cache so several GrapPA objectives can reuse the same cluster-to-truth match.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

from spine.cluster.quality import ClusterOverlapBatch, get_cluster_overlap_batch
from spine.data import ClusterLabelBatch, EdgeIndexBatch, IndexBatch

__all__ = [
    "ClusterOverlapCache",
    "ClusterQualityFilter",
    "OverlapThresholds",
]

ClusterOverlapCache = dict[str, ClusterOverlapBatch]
"""Overlap measurements keyed by the truth-instance field used for matching."""


class ClusterQualityFilter:
    """Apply reusable overlap-quality requirements to reconstructed objects.

    A reconstructed cluster is first matched to the truth instance with which
    it shares the most voxels. It is accepted only when that match satisfies
    every configured IoU, purity and efficiency requirement. Thresholds may be
    global scalars or arrays indexed by the target class of an object.

    The filter owns both threshold validation and the truth-instance field used
    for matching. A caller-provided :class:`ClusterOverlapCache` allows several
    objectives acting on the same clusters during one forward pass to share the
    comparatively expensive geometrical overlap calculation.

    Notes
    -----
    With no configured thresholds, every object passes and no truth matching is
    performed. Once filtering is active, clusters without a truth match fail
    the policy regardless of the numerical threshold values.
    """

    def __init__(
        self,
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        *,
        match_target: str = "group",
        num_classes: int | None = None,
        require_num_classes: bool = False,
    ) -> None:
        """Initialize the cluster-quality filter.

        Parameters
        ----------
        min_iou : float or sequence of float, optional
            Minimum intersection over union. A sequence supplies one value per
            target class.
        min_purity : float or sequence of float, optional
            Minimum fraction of a prediction owned by its truth match. A
            sequence supplies one value per target class.
        min_efficiency : float or sequence of float, optional
            Minimum fraction of the truth match recovered by a prediction. A
            sequence supplies one value per target class.
        match_target : str, default 'group'
            Field in ``data`` used to form truth instances, e.g. ``group`` or
            ``particle``.
        num_classes : int, optional
            Expected number of target classes and, consequently, the required
            length of class-dependent threshold sequences.
        require_num_classes : bool, default False
            Require ``num_classes`` during initialization whenever a sequence
            is configured. This is useful for regression objectives whose
            output width does not imply the number of quality classes.
        """
        self.match_target = match_target
        self.thresholds = OverlapThresholds(
            min_iou,
            min_purity,
            min_efficiency,
            num_classes=num_classes,
            require_num_classes=require_num_classes,
        )

    @property
    def active(self) -> bool:
        """Whether the filter has at least one overlap requirement."""
        return self.thresholds.active

    @property
    def class_dependent(self) -> bool:
        """Whether at least one requirement varies by target class."""
        return self.thresholds.class_dependent

    def validate_num_classes(self, num_classes: int) -> None:
        """Validate threshold-array lengths against a prediction width.

        Parameters
        ----------
        num_classes : int
            Number of classes represented by the prediction tensor.

        Raises
        ------
        ValueError
            If the class count is invalid or a threshold sequence has a
            different length.
        """
        self.thresholds.validate_num_classes(num_classes)

    def node_mask(
        self,
        data: ClusterLabelBatch,
        clusts: IndexBatch,
        classes: np.ndarray | None = None,
        cache: ClusterOverlapCache | None = None,
    ) -> np.ndarray:
        """Return the reconstructed objects satisfying the quality policy.

        Parameters
        ----------
        data : ClusterLabelBatch
            Structured voxel labels aligned with the cluster index space.
        clusts : IndexBatch
            Predicted cluster indexes grouped by batch entry.
        classes : np.ndarray, optional
            One target class per predicted cluster. Required only for
            class-dependent thresholds.
        cache : ClusterOverlapCache, optional
            Overlap results shared within one model forward pass.

        Returns
        -------
        np.ndarray
            Boolean validity mask with one entry per predicted cluster.

        Raises
        ------
        ValueError
            If class-dependent thresholds are configured but the class labels
            are missing, misaligned or outside the configured class range.
        """
        if not self.active:
            return np.ones(len(clusts.index_list), dtype=bool)

        # Cache only the geometrical overlap; each objective still applies its
        # own thresholds and, when requested, its own class-dependent values.
        overlap = self._get_overlap(data, clusts, cache)
        return self.thresholds.mask(overlap, classes)

    def edge_mask(
        self,
        data: ClusterLabelBatch,
        clusts: IndexBatch,
        edge_index: EdgeIndexBatch,
        classes: np.ndarray | None = None,
        cache: ClusterOverlapCache | None = None,
    ) -> np.ndarray:
        """Return edges whose two endpoints satisfy the quality policy.

        Parameters
        ----------
        data : ClusterLabelBatch
            Structured voxel labels aligned with the cluster index space.
        clusts : IndexBatch
            Predicted cluster indexes grouped by batch entry.
        edge_index : EdgeIndexBatch
            Batched edge endpoints indexing ``clusts``.
        classes : np.ndarray, optional
            One target class per edge. For class-dependent thresholds, the edge
            class selects the policy applied to both endpoint clusters.
        cache : ClusterOverlapCache, optional
            Overlap results shared within one model forward pass.

        Returns
        -------
        np.ndarray
            Boolean validity mask with one entry per edge.

        Raises
        ------
        IndexError
            If an endpoint does not index a predicted cluster.
        ValueError
            If class-dependent thresholds are configured but the edge classes
            are missing or do not align with the edge list.
        """
        # Validate endpoints even when filtering is inactive. An invalid graph
        # otherwise produces a deceptively well-formed all-true mask.
        edge_index_np = edge_index.to_numpy().index
        num_nodes = len(clusts.index_list)
        if np.any((edge_index_np < 0) | (edge_index_np >= num_nodes)):
            raise IndexError("Edge endpoints must index the node-quality mask.")
        if not self.active:
            return np.ones(edge_index_np.shape[1], dtype=bool)

        # Scalar requirements can be evaluated once per node and gathered at
        # the two endpoints of every edge.
        overlap = self._get_overlap(data, clusts, cache)
        if not self.class_dependent:
            node_mask = self.thresholds.mask(overlap)
            return np.all(node_mask[edge_index_np], axis=0)

        if classes is None:
            raise ValueError("Class-dependent edge thresholds require edge labels.")
        if len(classes) != edge_index_np.shape[1]:
            raise ValueError("Overlap threshold classes must align with edges.")

        # An edge class selects one policy, which must be satisfied by both of
        # that edge's endpoints. Nodes may therefore be treated differently in
        # distinct incident edges without recomputing geometrical overlaps.
        valid = np.zeros(len(classes), dtype=bool)
        for class_id in np.unique(classes):
            edge_selection = classes == class_id
            node_classes = np.full(num_nodes, class_id, dtype=np.int64)
            node_mask = self.thresholds.mask(overlap, node_classes)
            endpoints = edge_index_np[:, edge_selection]
            valid[edge_selection] = np.all(node_mask[endpoints], axis=0)

        return valid

    def _get_overlap(
        self,
        data: ClusterLabelBatch,
        clusts: IndexBatch,
        cache: ClusterOverlapCache | None,
    ) -> ClusterOverlapBatch:
        """Return overlap measurements, reusing a forward-local result.

        The cache key is the truth-instance field because all objectives in a
        GrapPA forward operate on the same predicted cluster collection. A
        standalone caller may omit the cache without changing the result.
        """
        overlap = None if cache is None else cache.get(self.match_target)
        if overlap is None:
            overlap = get_cluster_overlap_batch(data, clusts, self.match_target)

            # Populate only after a successful calculation, so a failed match
            # cannot leave a partial result for a subsequent objective.
            if cache is not None:
                cache[self.match_target] = overlap

        return overlap


class OverlapThresholds:
    """Validate and apply optional cluster-overlap quality thresholds.

    A threshold may be a scalar shared by every target class or a sequence
    containing one value per class. All three enabled requirements are combined
    with a logical AND. Sequence thresholds require a known class count so
    malformed configurations can fail during initialization.
    """

    def __init__(
        self,
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        num_classes: int | None = None,
        require_num_classes: bool = False,
    ) -> None:
        """Initialize overlap thresholds.

        Parameters
        ----------
        min_iou : float or sequence of float, optional
            Minimum intersection over union.
        min_purity : float or sequence of float, optional
            Minimum fraction of a prediction owned by its truth match.
        min_efficiency : float or sequence of float, optional
            Minimum fraction of the truth match recovered by a prediction.
        num_classes : int, optional
            Expected threshold-sequence length. Required whenever any
            threshold is class dependent and ``require_num_classes`` is set.
        require_num_classes : bool, default False
            Require the class count during initialization rather than allowing
            a caller to validate it once the prediction width is available.
        """
        if num_classes is not None and num_classes < 1:
            raise ValueError("Overlap threshold class count must be positive.")

        self.min_iou: float | np.ndarray | None = self._normalize(
            min_iou, "min_iou", num_classes, require_num_classes
        )
        self.min_purity: float | np.ndarray | None = self._normalize(
            min_purity, "min_purity", num_classes, require_num_classes
        )
        self.min_efficiency: float | np.ndarray | None = self._normalize(
            min_efficiency,
            "min_efficiency",
            num_classes,
            require_num_classes,
        )

    @property
    def active(self) -> bool:
        """Whether at least one overlap threshold is configured."""
        return any(
            threshold is not None
            for threshold in (self.min_iou, self.min_purity, self.min_efficiency)
        )

    @property
    def class_dependent(self) -> bool:
        """Whether any configured threshold varies by target class."""
        return any(
            isinstance(threshold, np.ndarray)
            for threshold in (self.min_iou, self.min_purity, self.min_efficiency)
        )

    def mask(
        self,
        overlap: ClusterOverlapBatch,
        classes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the objects satisfying every configured threshold.

        Parameters
        ----------
        overlap : ClusterOverlapBatch
            Best truth match and associated qualities for each predicted
            object.
        classes : np.ndarray, optional
            One target class per predicted object. Required when at least one
            threshold is class dependent.

        Returns
        -------
        np.ndarray
            Boolean validity mask with one entry per predicted object.

        Raises
        ------
        ValueError
            If class-dependent thresholds receive missing or invalid classes.
        """
        num_objects = len(overlap.match_ids.data)

        # An object without any truth intersection never defines a reliable
        # target, including when the numerical threshold itself is zero.
        valid = overlap.match_ids.data >= 0
        for values, threshold in (
            (overlap.ious.data, self.min_iou),
            (overlap.purities.data, self.min_purity),
            (overlap.efficiencies.data, self.min_efficiency),
        ):
            if threshold is None:
                continue

            if isinstance(threshold, np.ndarray):
                threshold_array = cast(np.ndarray, threshold)
                if classes is None:
                    raise ValueError(
                        "Class-dependent overlap thresholds require class labels."
                    )
                if len(classes) != num_objects:
                    raise ValueError(
                        "Overlap threshold classes must align with predicted objects."
                    )
                # Invalid or ignored classes fail the mask instead of indexing
                # from the end of the threshold array through NumPy semantics.
                class_valid = (classes >= 0) & (classes < len(threshold_array))
                valid &= class_valid
                selected = np.ones(num_objects, dtype=np.float32)
                selected[class_valid] = threshold_array[
                    classes[class_valid].astype(int)
                ]
                valid &= values >= selected
            else:
                valid &= values >= threshold

        return valid

    def validate_num_classes(self, num_classes: int) -> None:
        """Check all class-dependent thresholds against a class count.

        Parameters
        ----------
        num_classes : int
            Expected number of target classes.

        Raises
        ------
        ValueError
            If ``num_classes`` is not positive or a threshold sequence has a
            different length.
        """
        if num_classes < 1:
            raise ValueError("Overlap threshold class count must be positive.")
        for name, threshold in (
            ("min_iou", self.min_iou),
            ("min_purity", self.min_purity),
            ("min_efficiency", self.min_efficiency),
        ):
            if isinstance(threshold, np.ndarray):
                threshold_array = cast(np.ndarray, threshold)
                if len(threshold_array) != num_classes:
                    raise ValueError(
                        f"`{name}` must contain exactly {num_classes} values."
                    )

    @staticmethod
    def _normalize(
        threshold: float | Sequence[float] | None,
        name: str,
        num_classes: int | None,
        require_num_classes: bool,
    ) -> float | np.ndarray | None:
        """Normalize and validate one overlap threshold.

        Scalars remain scalars, while sequences become one-dimensional
        ``float32`` arrays. Every value must be finite and lie in ``[0, 1]``.
        """
        if threshold is None:
            return None
        if isinstance(threshold, (str, bytes)):
            raise TypeError(f"`{name}` must be numeric.")

        if np.isscalar(threshold):
            values = np.asarray([threshold], dtype=np.float32)
            normalized: float | np.ndarray = float(values[0])
        else:
            values = np.asarray(threshold, dtype=np.float32)
            if values.ndim != 1:
                raise ValueError(f"`{name}` must be a scalar or one-dimensional.")
            if num_classes is None and require_num_classes:
                raise ValueError(f"Class-dependent `{name}` requires `num_classes`.")
            if num_classes is not None and len(values) != num_classes:
                raise ValueError(f"`{name}` must contain exactly {num_classes} values.")
            normalized = values

        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"`{name}` values must lie in [0, 1].")
        return normalized
