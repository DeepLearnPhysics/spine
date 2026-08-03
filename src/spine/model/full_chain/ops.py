"""Shared object-aggregation operations for full-chain providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch_scatter import scatter_mean, scatter_std

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.grappa import GrapPA
from spine.utils.gnn.cluster import (
    get_cluster_label_batch,
    get_cluster_points_label_batch,
)
from spine.utils.gnn.evaluation import primary_assignment_batch
from spine.utils.ppn import ParticlePointPredictor

__all__ = ["AggregationOperations"]


class AggregationOperations:
    """Prepare GrapPA inputs and turn node assignments into voxel objects.

    This helper owns operations shared by particle- and interaction-level
    providers. It contains no trainable state; native GrapPA modules remain
    registered directly on the full-chain model.
    """

    def __init__(self, predict_points: dict[str, Any] | None = None) -> None:
        """Initialize the PPN-to-particle point post-processor.

        Parameters
        ----------
        predict_points : dict, optional
            Configuration forwarded to :class:`ParticlePointPredictor`.
        """
        self.point_predictor = ParticlePointPredictor(**(predict_points or {}))

    @staticmethod
    def restrict_clusters(
        clusts: IndexBatch,
        shapes: TensorBatch,
        accepted_shapes: Sequence[int],
    ) -> tuple[IndexBatch, TensorBatch, np.ndarray | None]:
        """Restrict clusters and aligned shapes to semantic classes.

        Parameters
        ----------
        clusts : IndexBatch
            Cluster voxel indexes.
        shapes : TensorBatch
            One semantic shape per cluster.
        accepted_shapes : sequence of int
            Shapes retained in the output.

        Returns
        -------
        IndexBatch
            Retained clusters.
        TensorBatch
            Shapes aligned with the retained clusters.
        numpy.ndarray or None
            Positions of retained clusters in the original cluster list, or
            ``None`` when no restriction was necessary.
        """
        # Preserve object identity in the common all-selected case. Consumers
        # use a null index to skip unnecessary scatter-back operations.
        shape_values = shapes.to_numpy().tensor
        mask = np.isin(shape_values, np.asarray(accepted_shapes))
        if np.all(mask):
            return clusts, shapes, None

        shape_index = np.flatnonzero(mask)
        cluster_list = np.asarray(clusts.index_list, dtype=object)[shape_index]
        restricted = IndexBatch(
            cluster_list,
            spans=clusts.spans,
            single_counts=clusts.single_counts[shape_index],
            batch_ids=clusts.batch_ids[shape_index],
            batch_size=clusts.batch_size,
            default=np.empty(0, dtype=np.int64),
        )
        restricted_shapes = TensorBatch(shape_values[shape_index], restricted.counts)
        return restricted, restricted_shapes, shape_index

    def prepare_grappa_input(
        self,
        model: GrapPA,
        state_outputs: dict[str, Any],
        data: TensorBatch,
        clusts: IndexBatch,
        shapes: TensorBatch,
        primaries: IndexBatch | None = None,
        clust_label: ClusterLabelBatch | None = None,
        coord_label: TensorBatch | None = None,
        point_use_primaries: bool = False,
    ) -> dict[str, TensorBatch | IndexBatch]:
        """Build explicit supplemental inputs required by GrapPA.

        Parameters
        ----------
        model : GrapPA
            Native GrapPA model whose node encoder defines required features.
        state_outputs : dict
            Public outputs from preceding stages, including optional PPN
            predictions.
        data : TensorBatch
            Canonical sparse voxel data.
        clusts : IndexBatch
            Node-defining voxel clusters.
        shapes : TensorBatch
            Semantic shape of each node.
        primaries : IndexBatch, optional
            Primary-fragment indexes used as point references.
        clust_label : ClusterLabelBatch, optional
            Structured truth used to derive point labels when PPN output is
            unavailable.
        coord_label : TensorBatch, optional
            Particle start and end point truth.
        point_use_primaries : bool, default False
            Associate point features with primary fragments rather than full
            node clusters.

        Returns
        -------
        dict
            Keyword arguments accepted by :class:`GrapPA`.

        Raises
        ------
        ValueError
            If the configured encoder requires unavailable point inputs or
            primary indexes.
        """
        result: dict[str, TensorBatch | IndexBatch] = {
            "data": data,
            "clusts": clusts,
            "shapes": shapes,
        }
        encoder = model.node_encoder

        # Reconstructed chains must explicitly associate predicted PPN points
        # with their current objects; truth-only runs can use label endpoints.
        if getattr(encoder, "add_points", False):
            reference = clusts
            if point_use_primaries:
                if primaries is None:
                    raise ValueError("Primary-based points require `primaries`.")
                reference = primaries

            if "ppn_points" in state_outputs:
                result["points"] = self.point_predictor(
                    data,
                    reference,
                    shapes,
                    state_outputs["ppn_points"],
                )
            else:
                if coord_label is None or clust_label is None:
                    raise ValueError(
                        "GrapPA point features require `ppn_points` or both "
                        "`coord_label` and `clust_label`."
                    )
                result["points"] = get_cluster_points_label_batch(
                    clust_label,
                    coord_label,
                    reference,
                    random_order=bool(getattr(encoder, "random_order", False)),
                )

        elif coord_label is not None:
            result["coord_label"] = coord_label

        # Shape cannot be inferred from feature-only canonical data, so build
        # all requested supplemental values here in one aligned tensor.
        add_value = bool(getattr(encoder, "add_value", False))
        add_shape = bool(getattr(encoder, "add_shape", False))
        if add_value or add_shape:
            extra = []

            # Charge summaries use the cluster-to-voxel membership map.
            if add_value:
                values = data.values.torch_tensor()[clusts.full_index]
                index_ids = torch.as_tensor(
                    clusts.index_ids,
                    dtype=torch.long,
                    device=data.device,
                )
                extra.extend(
                    (scatter_mean(values, index_ids), scatter_std(values, index_ids))
                )
            # Semantic shape is already defined per cluster.
            if add_shape:
                extra.append(
                    torch.as_tensor(
                        shapes.to_numpy().tensor,
                        dtype=data.dtype,
                        device=data.device,
                    )
                )
            result["extra"] = TensorBatch(torch.stack(extra).t(), clusts.counts)

        return result

    @staticmethod
    def build_groups(
        clusts: IndexBatch,
        shapes: TensorBatch,
        assignments: TensorBatch,
        primary_mask: TensorBatch | np.ndarray | None = None,
        aggregate_shapes: bool = False,
        shape_use_primary: bool = False,
        retain_primaries: bool = False,
    ) -> tuple[IndexBatch, TensorBatch, IndexBatch]:
        """Merge clusters according to per-node group assignments.

        Parameters
        ----------
        clusts : IndexBatch
            Input node clusters.
        shapes : TensorBatch
            Semantic shape per input node.
        assignments : TensorBatch
            Event-local group identifier per input node.
        primary_mask : TensorBatch or numpy.ndarray, optional
            Boolean marker identifying primary nodes.
        aggregate_shapes : bool, default False
            Produce one semantic shape per merged group.
        shape_use_primary : bool, default False
            Take the group shape from its primary node when available instead
            of using the modal member shape.
        retain_primaries : bool, default False
            Return the primary member voxels for each group.

        Returns
        -------
        IndexBatch
            Merged voxel groups.
        TensorBatch
            Group-level semantic shapes, empty when ``aggregate_shapes`` is
            disabled.
        IndexBatch
            Primary-member voxel indexes, or the full groups when primary
            retention is disabled.
        """
        # Convert aligned inputs once before iterating event boundaries.
        assignments_np = assignments.to_numpy()
        shapes_np = shapes.to_numpy()
        primary_np = (
            primary_mask.to_numpy()
            if isinstance(primary_mask, TensorBatch)
            else primary_mask
        )
        groups: list[np.ndarray] = []
        group_shapes: list[int] = []
        group_primaries: list[np.ndarray] = []
        counts: list[int] = []
        single_counts: list[int] = []
        primary_counts: list[int] = []

        # Group assignments are event-local. Convert each resulting voxel set
        # back to the globally offset index namespace used by IndexBatch.
        for batch_id in range(assignments.batch_size):
            clusts_b = clusts[batch_id]
            assignment_b = assignments_np[batch_id]
            shape_b = shapes_np[batch_id]
            primary_b = None if primary_np is None else primary_np[batch_id]
            group_ids = np.unique(assignment_b)
            counts.append(len(group_ids))

            for group_id in group_ids:
                members = np.flatnonzero(assignment_b == group_id)
                member_clusts = [clusts_b[index] for index in members]
                group = clusts.offsets[batch_id] + np.concatenate(member_clusts)
                groups.append(group)
                single_counts.append(len(group))

                # Select at most one primary node. Missing primaries fall back
                # to group-level information below.
                primary_id = None
                if primary_b is not None:
                    candidates = members[np.asarray(primary_b[members], dtype=bool)]
                    if len(candidates):
                        primary_id = int(candidates[0])

                # Determine the semantic identity of the merged object.
                if aggregate_shapes:
                    if shape_use_primary and primary_id is not None:
                        group_shapes.append(int(shape_b[primary_id]))
                    else:
                        values, frequencies = np.unique(
                            shape_b[members],
                            return_counts=True,
                        )
                        group_shapes.append(int(values[np.argmax(frequencies)]))

                # Keep the primary node's voxels for downstream point features.
                if retain_primaries:
                    primary = (
                        group
                        if primary_id is None
                        else (clusts.offsets[batch_id] + clusts_b[primary_id])
                    )
                    group_primaries.append(primary)
                    primary_counts.append(len(primary))

        # Rebuild batched products from the flattened event-local lists.
        group_batch = IndexBatch(
            groups,
            clusts.spans,
            counts,
            single_counts,
            default=np.empty(0, dtype=np.int64),
        )
        shape_counts = (
            counts if aggregate_shapes else np.zeros(clusts.batch_size, dtype=np.int64)
        )
        shape_batch = TensorBatch(
            np.asarray(group_shapes, dtype=np.int64),
            shape_counts,
        )
        if retain_primaries:
            primary_batch = IndexBatch(
                group_primaries,
                clusts.spans,
                counts,
                primary_counts,
                default=np.empty(0, dtype=np.int64),
            )
        else:
            primary_batch = group_batch
        return group_batch, shape_batch, primary_batch

    def run_grappa(
        self,
        model: GrapPA,
        state_outputs: dict[str, Any],
        data: TensorBatch,
        clusts: IndexBatch,
        shapes: TensorBatch,
        accepted_shapes: Sequence[int],
        primaries: IndexBatch | None = None,
        clust_label: ClusterLabelBatch | None = None,
        coord_label: TensorBatch | None = None,
        aggregate_shapes: bool = False,
        shape_use_primary: bool = False,
        point_use_primary: bool = False,
        retain_primaries: bool = False,
    ) -> tuple[IndexBatch, TensorBatch, IndexBatch, np.ndarray | None, dict[str, Any]]:
        """Execute GrapPA and build voxel groups from its predictions.

        Parameters
        ----------
        model : GrapPA
            Native graph model.
        state_outputs : dict
            Public outputs from preceding chain stages.
        data : TensorBatch
            Canonical sparse voxel data.
        clusts : IndexBatch
            Candidate graph nodes.
        shapes : TensorBatch
            Semantic shape per graph node.
        accepted_shapes : sequence of int
            Shapes owned by this graph path.
        primaries : IndexBatch, optional
            Primary node voxels used for point construction.
        clust_label : ClusterLabelBatch, optional
            Structured cluster truth.
        coord_label : TensorBatch, optional
            Particle coordinate truth.
        aggregate_shapes : bool, default False
            Build one semantic shape per predicted group.
        shape_use_primary : bool, default False
            Infer group shape from its predicted primary.
        point_use_primary : bool, default False
            Use primary voxels to associate PPN points.
        retain_primaries : bool, default False
            Retain primary voxel indexes in the grouped output.

        Returns
        -------
        IndexBatch
            Predicted voxel groups.
        TensorBatch
            Predicted group shapes.
        IndexBatch
            Predicted primary-member voxel indexes.
        numpy.ndarray or None
            Positions of retained nodes in the unrestricted input.
        dict
            Native GrapPA outputs.
        """
        # Restrict the graph to the semantic classes owned by this path.
        clusts, shapes, shape_index = self.restrict_clusters(
            clusts,
            shapes,
            accepted_shapes,
        )
        model_input = self.prepare_grappa_input(
            model,
            state_outputs,
            data,
            clusts,
            shapes,
            primaries,
            clust_label,
            coord_label,
            point_use_primary,
        )
        # Run the native model without changing its standalone interface.
        output = model(**model_input)

        # Primary-aware grouping converts node logits to one primary marker per
        # predicted group before merging voxel indexes.
        primary_mask = None
        if shape_use_primary or retain_primaries:
            if "node_pred" not in output:
                raise ValueError("Primary-aware grouping requires `node_pred`.")
            primary_mask = primary_assignment_batch(
                output["node_pred"].to_numpy(),
                output["group_pred"],
            )
        groups, group_shapes, group_primaries = self.build_groups(
            clusts,
            shapes,
            output["group_pred"],
            primary_mask,
            aggregate_shapes,
            shape_use_primary,
            retain_primaries,
        )
        return groups, group_shapes, group_primaries, shape_index, output

    @classmethod
    def group_labels(
        cls,
        clust_label: ClusterLabelBatch,
        clusts: IndexBatch,
        shapes: TensorBatch,
        accepted_shapes: Sequence[int] | None = None,
        aggregate_shapes: bool = False,
        shape_use_primary: bool = False,
        retain_primaries: bool = False,
    ) -> tuple[IndexBatch, TensorBatch, IndexBatch, np.ndarray | None]:
        """Aggregate clusters using truth group and primary assignments.

        Parameters
        ----------
        clust_label : ClusterLabelBatch
            Structured voxel and particle truth.
        clusts : IndexBatch
            Input node clusters.
        shapes : TensorBatch
            Semantic shape per input node.
        accepted_shapes : sequence of int, optional
            Shapes retained before grouping.
        aggregate_shapes : bool, default False
            Produce one shape per truth group.
        shape_use_primary : bool, default False
            Take the group shape from its truth primary.
        retain_primaries : bool, default False
            Retain truth-primary voxel indexes.

        Returns
        -------
        IndexBatch
            Truth-grouped voxel indexes.
        TensorBatch
            Group-level semantic shapes.
        IndexBatch
            Truth-primary voxel indexes.
        numpy.ndarray or None
            Positions of retained nodes in the original input.
        """
        # Apply the same semantic ownership restriction used by learned paths.
        shape_index = None
        if accepted_shapes is not None:
            clusts, shapes, shape_index = cls.restrict_clusters(
                clusts,
                shapes,
                accepted_shapes,
            )
        # Reduce voxel truth to one group and optional primary assignment per
        # input cluster before reusing the common merger.
        group_ids = get_cluster_label_batch(clust_label, clusts, "group")
        primary_mask = None
        if shape_use_primary or retain_primaries:
            primary_mask = get_cluster_label_batch(
                clust_label,
                clusts,
                "group_primary",
            )
        return (
            *cls.build_groups(
                clusts,
                shapes,
                group_ids,
                primary_mask,
                aggregate_shapes,
                shape_use_primary,
                retain_primaries,
            ),
            shape_index,
        )
