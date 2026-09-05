"""Graph-level augmentations for GrapPA training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spine.cluster.label import get_cluster_label_batch
from spine.constants.factory import enum_factory
from spine.data import ClusterLabelBatch, EdgeIndexBatch, IndexBatch, TensorBatch
from spine.utils.conditional import torch

__all__ = ["EdgeDropout", "EdgeSelection", "NodeDropout", "NodeSelection"]


class _BatchSelection:
    """Reusable event-aware selection over one graph product axis."""

    axis = "element"

    def __init__(self, keep: TensorBatch) -> None:
        values = keep.to_numpy().data
        if values.ndim != 1:
            raise ValueError(
                f"{self.axis.capitalize()} selection must be a one-dimensional mask."
            )

        self.keep = keep
        self.mask = values.astype(bool, copy=False)
        self.counts = np.zeros(keep.batch_size, dtype=np.int64)
        edges = keep.to_numpy().edges
        for batch_id in range(keep.batch_size):
            lower, upper = edges[batch_id : batch_id + 2]
            self.counts[batch_id] = np.count_nonzero(self.mask[lower:upper])

    def filter_tensor(self, batch: TensorBatch) -> TensorBatch:
        """Apply the selection to an aligned tensor batch.

        Parameters
        ----------
        batch : TensorBatch
            Features, targets or validity values aligned with the original
            graph-product axis.

        Returns
        -------
        TensorBatch
            Selected values with recomputed event counts and retained schema.
        """
        self.validate(batch.counts, batch.shape[0], "tensor batch")
        backend_mask = self._backend_mask(batch.is_numpy, batch.device)

        return TensorBatch(
            batch.data[backend_mask],
            self.counts,
            has_batch_col=batch.has_batch_col,
            coord_cols=batch.coord_cols,
            schema=batch.schema,
            meta=batch.meta,
        )

    def _backend_mask(self, is_numpy: bool, device: Any) -> Any:
        """Return the mask on the backend and device of its target."""
        if is_numpy:
            return self.mask
        return torch.as_tensor(self.mask, device=device)

    def validate(self, counts: Any, size: int, target: str) -> None:
        """Check that a target retains the original event partition.

        Parameters
        ----------
        counts : array-like
            Per-event counts carried by the target product.
        size : int
            Total number of target elements.
        target : str
            Human-readable product name used in alignment errors.
        """
        if not isinstance(counts, np.ndarray):
            counts = counts.detach().cpu().numpy()
        keep_counts = self.keep.counts
        if not isinstance(keep_counts, np.ndarray):
            keep_counts = keep_counts.detach().cpu().numpy()
        if size != self.keep.shape[0] or not np.array_equal(counts, keep_counts):
            raise ValueError(
                f"{self.axis.capitalize()} selection must align with the {target}."
            )


class EdgeSelection(_BatchSelection):
    """Reusable event-aware selection over an original graph edge axis.

    The selection owns the bookkeeping needed to apply one sampled graph
    perturbation consistently to edge indexes, materialized features and
    cached supervision.

    Parameters
    ----------
    keep : TensorBatch
        One-dimensional Boolean mask partitioned like the original graph.
    """

    axis = "edge"

    def filter_edge_index(self, edge_index: EdgeIndexBatch) -> EdgeIndexBatch:
        """Apply the selection to a graph incidence matrix.

        Parameters
        ----------
        edge_index : EdgeIndexBatch
            Graph whose edge axis matches the original selection.

        Returns
        -------
        EdgeIndexBatch
            Graph containing only retained edges, with updated event counts.
        """
        self.validate(edge_index.counts, edge_index.shape[1], "edge index")
        backend_mask = self._backend_mask(edge_index.is_numpy, edge_index.device)

        return EdgeIndexBatch(
            edge_index.index[:, backend_mask],
            self.counts,
            edge_index.spans,
            edge_index.directed,
        )

    def compose(self, selection: "EdgeSelection") -> "EdgeSelection":
        """Compose a second selection defined on the retained edge axis.

        The returned mask is aligned with this selection's original graph,
        which lets cached supervision be filtered once after several graph
        augmentations have been applied in sequence.

        Parameters
        ----------
        selection : EdgeSelection
            Selection over the edges retained by this selection.

        Returns
        -------
        EdgeSelection
            Combined original-to-final edge selection.
        """
        selection.validate(self.counts, int(np.sum(self.counts)), "edge selection")
        combined = self.mask.copy()
        combined[self.mask] = selection.mask

        return EdgeSelection(TensorBatch(combined, self.keep.counts))


class NodeSelection(_BatchSelection):
    """Reusable event-aware selection over an original graph node axis.

    Besides filtering node-aligned tensor and cluster products, the selection
    removes incident edges and remaps their endpoints into the compact node
    namespace of the augmented graph.

    Parameters
    ----------
    keep : TensorBatch
        One-dimensional Boolean mask partitioned like the original nodes.
    """

    axis = "node"

    def filter_index(self, batch: IndexBatch) -> IndexBatch:
        """Apply the selection to a node-aligned cluster index batch.

        Parameters
        ----------
        batch : IndexBatch
            Cluster membership, with one flat or jagged member per node.

        Returns
        -------
        IndexBatch
            Membership for the retained nodes. Parent voxel spans and the
            underlying NumPy/Torch representation are preserved.
        """
        self.validate(batch.counts, len(batch.data), "index batch")
        backend_mask = self._backend_mask(batch.is_numpy, batch.device)

        if batch.is_list:
            indexes = [
                index for index, keep in zip(batch.index_list, self.mask) if keep
            ]
            single_counts = batch.single_counts[backend_mask]
            return IndexBatch(indexes, batch.spans, self.counts, single_counts)

        return IndexBatch(batch.index[backend_mask], batch.spans, self.counts)

    def filter_edge_index(
        self, edge_index: EdgeIndexBatch
    ) -> tuple[EdgeIndexBatch, EdgeSelection]:
        """Remove incident edges and compact retained node indexes.

        Parameters
        ----------
        edge_index : EdgeIndexBatch
            Graph whose per-event node spans match the original selection.

        Returns
        -------
        EdgeIndexBatch
            Graph over the compact retained-node namespace.
        EdgeSelection
            Selection over the input graph's edge axis. This can be applied
            to edge features and supervision associated with that graph.
        """
        self.validate(edge_index.spans, int(np.sum(self.keep.counts)), "edge index")
        numpy_index = edge_index.to_numpy().index

        # Build one global old-to-new node map while retaining event boundaries.
        node_map = np.full(len(self.mask), -1, dtype=np.int64)
        old_lower = 0
        new_lower = 0
        for old_count, new_count in zip(self.keep.to_numpy().counts, self.counts):
            old_upper = old_lower + int(old_count)
            local_keep = np.flatnonzero(self.mask[old_lower:old_upper])
            node_map[old_lower + local_keep] = new_lower + np.arange(len(local_keep))
            old_lower = old_upper
            new_lower += int(new_count)

        edge_keep = self.mask[numpy_index[0]] & self.mask[numpy_index[1]]
        remapped = node_map[numpy_index[:, edge_keep]]
        if not edge_index.is_numpy:
            remapped = torch.as_tensor(
                remapped, dtype=edge_index.index.dtype, device=edge_index.device
            )

        edge_selection = EdgeSelection(TensorBatch(edge_keep, edge_index.counts))
        filtered = EdgeIndexBatch(
            remapped,
            edge_selection.counts,
            self.counts,
            edge_index.directed,
        )
        return filtered, edge_selection


class EdgeDropout:
    """Randomly remove graph edges during training.

    Directed edges are sampled independently. Undirected GrapPA graphs store
    each connection as adjacent reciprocal edges, so one decision is sampled
    per pair and applied to both directions. This preserves the graph's
    undirected contract while allowing entire connections to disappear.

    Parameters
    ----------
    probability : float
        Probability of dropping each directed edge or reciprocal edge pair.
        Must lie in the closed interval ``[0, 1]``.
    """

    def __init__(self, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Edge dropout probability must be between 0 and 1.")

        self.probability = float(probability)

    def __call__(self, edge_index: EdgeIndexBatch) -> EdgeSelection:
        """Sample an event-aware edge selection.

        Parameters
        ----------
        edge_index : EdgeIndexBatch
            Graph incidence matrix before augmentation.

        Returns
        -------
        EdgeSelection
            Selection aligned with the original edge axis. It can be applied
            consistently to graph indexes, materialized features and cached
            supervision.

        Raises
        ------
        ValueError
            If an undirected event does not contain adjacent reciprocal pairs.
        """
        numpy_index = edge_index.to_numpy()
        index = numpy_index.index
        counts = numpy_index.counts
        keep = np.zeros(index.shape[1], dtype=bool)

        lower = 0
        for count_value in counts:
            count = int(count_value)
            upper = lower + count
            if edge_index.directed:
                event_keep = np.random.random(count) >= self.probability
            else:
                event_keep = self._sample_undirected(index[:, lower:upper])

            keep[lower:upper] = event_keep
            lower = upper

        return EdgeSelection(TensorBatch(keep, counts))

    def _sample_undirected(self, index: np.ndarray) -> np.ndarray:
        """Sample adjacent reciprocal pairs from one undirected event.

        Parameters
        ----------
        index : np.ndarray
            ``(2, E)`` event-local incidence matrix.

        Returns
        -------
        np.ndarray
            Boolean edge selection in which both directions of every pair
            receive the same decision.
        """
        num_edges = index.shape[1]
        if num_edges % 2:
            raise ValueError(
                "Undirected edge dropout requires an even edge count per event."
            )

        # GraphBase guarantees this layout; validate materialized inputs too.
        if num_edges and not np.array_equal(index[:, 1::2], index[::-1, ::2]):
            raise ValueError(
                "Undirected edge dropout requires adjacent reciprocal edge pairs."
            )

        pair_keep = np.random.random(num_edges // 2) >= self.probability
        return np.repeat(pair_keep, 2)


class NodeDropout:
    """Randomly remove individual nodes or complete physical node groups.

    Sampling is performed independently in every event. When ``group_by`` is
    configured, all nodes sharing a nonnegative group label receive the same
    decision. Nodes with invalid negative labels are retained, rather than
    accidentally treating all invalid objects as one large group.

    Parameters
    ----------
    probability : float
        Probability of dropping each node or complete node group. Must lie in
        the closed interval ``[0, 1]``.
    group_by : str, optional
        Cluster-label field used by ``GrapPA`` to derive group IDs for
        live inputs. If omitted, nodes are sampled independently.
    select : mapping, optional
        Static label selection which limits the nodes or groups eligible for
        dropout. Values within one field are OR-ed and separate fields are
        AND-ed. ``shape`` and ``pid`` values accept canonical string names;
        ``primary`` is accepted as an alias for ``group_primary``.
    group_match : str, default 'any'
        For grouped dropout, whether ``"any"`` or ``"all"`` nodes in a group
        must pass ``select`` before the complete group becomes eligible.
    keep_one : bool, default True
        If `True`, retain at least one sampled node/group in every nonempty
        event. Invalid group labels are not considered sampled groups.
    """

    def __init__(
        self,
        probability: float,
        group_by: str | None = None,
        select: Mapping[str, int | str | Sequence[int | str]] | None = None,
        group_match: str = "any",
        keep_one: bool = True,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Node dropout probability must be between 0 and 1.")
        if group_by is not None and not group_by:
            raise ValueError("Node dropout `group_by` must be a nonempty field name.")
        if group_match not in ("any", "all"):
            raise ValueError("Node dropout `group_match` must be 'any' or 'all'.")
        if group_by is None and group_match != "any":
            raise ValueError("Node dropout `group_match` requires grouped dropout.")

        self.probability = float(probability)
        self.group_by = group_by
        self.select = self._normalize_select(select)
        self.group_match = group_match
        self.keep_one = keep_one

    def __call__(
        self,
        counts: Any,
        group_ids: TensorBatch | None = None,
        eligible: TensorBatch | None = None,
    ) -> NodeSelection:
        """Sample an event-aware node selection.

        Parameters
        ----------
        counts : sequence of int or array-like
            Number of graph nodes in each event.
        group_ids : TensorBatch, optional
            Node-aligned group labels. Required when ``group_by`` is set and
            ignored for independent node dropout.
        eligible : TensorBatch, optional
            Static node-aligned eligibility mask. Required when ``select`` is
            configured. Ineligible nodes and groups are always retained.

        Returns
        -------
        NodeSelection
            Selection aligned with the original node axis.

        Raises
        ------
        ValueError
            If grouped dropout lacks aligned group IDs.
        """
        counts_array = self._counts_array(counts)
        num_nodes = int(np.sum(counts_array))
        groups = None
        if self.group_by is not None:
            if group_ids is None:
                raise ValueError(
                    "Grouped node dropout requires node-aligned "
                    "`node_dropout_group_ids`."
                )
            groups = group_ids.to_numpy()
            if (
                groups.data.ndim != 1
                or groups.shape[0] != num_nodes
                or not np.array_equal(groups.counts, counts_array)
            ):
                raise ValueError("Node dropout group IDs must align with graph nodes.")

        eligible_mask = np.ones(num_nodes, dtype=bool)
        if self.select is not None:
            if eligible is None:
                raise ValueError(
                    "Selected node dropout requires node-aligned "
                    "`node_dropout_eligible`."
                )
            eligible_np = eligible.to_numpy()
            if (
                eligible_np.data.ndim != 1
                or eligible_np.shape[0] != num_nodes
                or not np.array_equal(eligible_np.counts, counts_array)
            ):
                raise ValueError(
                    "Node dropout eligibility must align with graph nodes."
                )
            eligible_mask = eligible_np.data.astype(bool, copy=False)

        keep = np.ones(num_nodes, dtype=bool)
        lower = 0
        for count in counts_array:
            upper = lower + int(count)
            if groups is None:
                keep[lower:upper] = self._sample_units(eligible_mask[lower:upper])
            else:
                keep[lower:upper] = self._sample_groups(
                    groups.data[lower:upper], eligible_mask[lower:upper]
                )
            lower = upper

        return NodeSelection(TensorBatch(keep, counts_array))

    def build_eligibility(
        self,
        data: ClusterLabelBatch,
        clusts: IndexBatch,
    ) -> TensorBatch:
        """Build the configured static eligibility mask from live labels.

        Parameters
        ----------
        data : ClusterLabelBatch
            Structured voxel and particle labels.
        clusts : IndexBatch
            Cluster membership defining the graph nodes.

        Returns
        -------
        TensorBatch
            Boolean node-aligned mask satisfying every configured label field.

        Raises
        ------
        ValueError
            If no static label selection was configured.
        """
        if self.select is None:
            raise ValueError("Cannot build eligibility without a `select` mapping.")

        eligible = np.ones(len(clusts.data), dtype=bool)
        for field, accepted in self.select.items():
            labels = get_cluster_label_batch(data, clusts, field).to_numpy().data
            eligible &= np.isin(labels, accepted)

        return TensorBatch(eligible, clusts.counts)

    def _sample_units(self, eligible: np.ndarray) -> np.ndarray:
        """Sample eligible independent node decisions for one event."""
        candidate_ids = np.flatnonzero(eligible)
        keep = np.ones(len(eligible), dtype=bool)
        keep[candidate_ids] = np.random.random(len(candidate_ids)) >= self.probability
        if self.keep_one and len(candidate_ids) > 0 and not np.any(keep[candidate_ids]):
            keep[candidate_ids[np.random.randint(len(candidate_ids))]] = True
        return keep

    def _sample_groups(
        self,
        group_ids: np.ndarray,
        eligible: np.ndarray,
    ) -> np.ndarray:
        """Sample one shared decision per eligible valid group in one event."""
        group_ids = np.asarray(group_ids, dtype=np.int64)
        keep = np.ones(len(group_ids), dtype=bool)
        valid_groups = np.unique(group_ids[group_ids >= 0])
        candidate_groups = []
        for group_id in valid_groups:
            group_eligible = eligible[group_ids == group_id]
            matches = (
                np.any(group_eligible)
                if self.group_match == "any"
                else np.all(group_eligible)
            )
            if matches:
                candidate_groups.append(group_id)

        group_keep = np.random.random(len(candidate_groups)) >= self.probability
        if self.keep_one and candidate_groups and not np.any(group_keep):
            group_keep[np.random.randint(len(candidate_groups))] = True

        # Assign shared decisions only to candidate groups. Invalid labels and
        # groups outside the static selection remain untouched.
        for group_id, retain in zip(candidate_groups, group_keep):
            keep[group_ids == group_id] = retain
        return keep

    @staticmethod
    def _normalize_select(
        select: Mapping[str, int | str | Sequence[int | str]] | None,
    ) -> dict[str, np.ndarray] | None:
        """Normalize static label selectors to canonical fields and integers."""
        if select is None:
            return None
        if not select:
            raise ValueError("Node dropout `select` must not be empty.")

        aliases = {"primary": "group_primary", "type": "pid"}
        normalized = {}
        for raw_field, raw_values in select.items():
            field = aliases.get(raw_field, raw_field)
            values = (
                [raw_values]
                if isinstance(raw_values, (str, bytes)) or np.isscalar(raw_values)
                else list(raw_values)
            )
            if not values:
                raise ValueError(
                    f"Node dropout selection for `{raw_field}` must not be empty."
                )

            parsed = []
            for value in values:
                if isinstance(value, str):
                    if field not in ("shape", "pid"):
                        raise ValueError(
                            f"String values are not supported for node label "
                            f"field `{raw_field}`."
                        )
                    value = enum_factory(field, value)
                parsed.append(int(value))
            normalized[field] = np.asarray(parsed, dtype=np.int64)

        return normalized

    @staticmethod
    def _counts_array(counts: Any) -> np.ndarray:
        """Normalize event counts to a one-dimensional NumPy integer array."""
        if not isinstance(counts, np.ndarray):
            if hasattr(counts, "detach"):
                counts = counts.detach().cpu().numpy()
            else:
                counts = np.asarray(counts)
        counts = np.asarray(counts, dtype=np.int64)
        if counts.ndim != 1 or np.any(counts < 0):
            raise ValueError("Node counts must be a nonnegative one-dimensional array.")
        return counts
