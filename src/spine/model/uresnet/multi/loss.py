"""Joint supervision for configurable shared UResNet reconstruction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from spine.data import ClusterLabelBatch, TensorBatch
from spine.model.common.loss_balancing import LossBalancer, LossTerm
from spine.model.graph_spice.model import GraphSPICELoss
from spine.model.uresnet.ppn import UResNetPPNLoss

from .semantics import FragmentSemanticLoss, resolve_semantics

__all__ = ["UResNetMultiLoss"]


class UResNetMultiLoss(torch.nn.Module):
    """Supervise voxel semantics, point proposals, edges and fragment classes.

    Original cluster IDs and semantic truth define the edge targets, independently
    of predicted routing. Learned fragment classification adds a categorical
    objective over predicted components. The voxel segmentation objective remains
    an auxiliary loss in fragment-first mode and can be assigned a zero priority
    through fixed loss balancing when a fragment-only semantic objective is wanted.
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        ppn: dict[str, Any],
        graph_spice: dict[str, Any],
        uresnet_loss: dict[str, Any],
        ppn_loss: dict[str, Any],
        graph_spice_loss: dict[str, Any],
        vertex: dict[str, Any] | None = None,
        vertex_loss: dict[str, Any] | None = None,
        proposal_decoder: dict[str, Any] | None = None,
        semantic_coupling: str = "hard",
        fragment_semantics: str | None = None,
        semantic_conditioning: dict[str, Any] | None = None,
        loss_balancing: dict[str, Any] | None = None,
    ) -> None:
        """Initialize native objectives and their joint balancing policy.

        Parameters
        ----------
        uresnet : dict
            Shared backbone and voxel semantic configuration.
        ppn : dict
            Particle-point proposal configuration.
        graph_spice : dict
            Embedding and graph configuration used by the network.
        uresnet_loss : dict
            Voxel segmentation loss configuration.
        ppn_loss : dict
            Point-proposal loss configuration.
        graph_spice_loss : dict
            Native binary edge loss and optional clustering metric settings.
        vertex : dict, optional
            Additional interaction-vertex proposal configuration.
        vertex_loss : dict, optional
            Objective settings for the optional vertex branch.
        proposal_decoder : dict, optional
            Shared particle/vertex proposal decoder settings.
        semantic_coupling : {'hard', 'independent', 'fragment_first', 'soft'}
            Network coupling mode, used to match its graph partitioning.
        fragment_semantics : {'routed', 'pooled', 'learned'}, optional
            Final classification policy. Learned semantics add a supervised head.
        semantic_conditioning : dict, optional
            Network-owned soft conditioning settings, accepted for the shared
            model/loss configuration contract.
        loss_balancing : dict, optional
            Sum, fixed or uncertainty balancing. Objective names include the
            existing segmentation/PPN leaves, ``fragmentation``, and optionally
            ``fragment_semantics`` for learned fragment classification.
        """
        # Resolve the same architectural policy as the network constructor
        super().__init__()
        del semantic_conditioning
        policy = resolve_semantics(semantic_coupling, fragment_semantics)
        self.class_agnostic = semantic_coupling != "hard"
        self.num_classes = uresnet["num_classes"]

        # Reuse the native voxel and proposal objectives without double balancing
        self.uresnet_ppn = UResNetPPNLoss(
            uresnet, uresnet_loss, ppn, ppn_loss, vertex, vertex_loss, proposal_decoder
        )

        # Clustering metrics must use the same single partition as agnostic graphs.
        # The internal partition identifier does not represent a semantic class.
        config = deepcopy(graph_spice)
        if self.class_agnostic:
            shapes = config.get("shapes", ["shower", "track", "michel", "delta"])
            if not shapes:
                raise ValueError("At least one fragment shape must be configured.")
            config["shapes"] = shapes[:1]
        self.graph_spice = GraphSPICELoss(config, graph_spice_loss)
        self.fragment_loss = FragmentSemanticLoss() if policy == "learned" else None

        # Edge objectives can combine Dice and cross entropy; learned fragment
        # classification instead has a categorical soft-target interpretation.
        families = dict(self.uresnet_ppn.loss_balancer.families)
        families["fragmentation"] = "composite"
        if self.fragment_loss is not None:
            families["fragment_semantics"] = "categorical"
        self.loss_balancer = LossBalancer(families, loss_balancing)

    def label_graph(
        self,
        seg_label: TensorBatch,
        clust_label: ClusterLabelBatch,
        graph: dict[str, Any],
    ) -> None:
        """Attach coherent edge targets in the graph's filtered voxel domain.

        Parameters
        ----------
        seg_label : TensorBatch
            Original voxel semantic labels, before prediction-based adaptation.
        clust_label : ClusterLabelBatch
            Original cluster IDs aligned to the same voxel rows.
        graph : dict
            Namespaced graph outputs after removing their public prefix. The
            dictionary is updated with edge and node targets for loss/metrics.

        Notes
        -----
        A positive edge requires equal nonnegative cluster IDs and equal valid
        truth semantic classes. Unassigned cluster IDs never connect merely
        because both endpoints contain the same negative sentinel.
        """
        # Align supervision with the exact nodes retained by the graph head
        semantics, clusters = self.graph_spice.filter_class(
            seg_label, clust_label, graph["filter_index"]
        )
        classes = semantics.values.torch_tensor().long().flatten()
        ids = clusters.voxel_field(self.graph_spice.target_col).torch_tensor().long()
        ids = ids.flatten()

        # Reuse graph-index validation and native cluster equality targets
        labels = self.graph_spice.get_edge_labels(
            clusters, graph["edge_index"], graph["node_clusts"], graph["edge_clusts"]
        )
        values = labels.torch_tensor()
        edges = graph["edge_index"].torch_tensor().long()
        for nodes, edge_group in zip(
            graph["node_clusts"].index_list, graph["edge_clusts"].index_list
        ):
            endpoints = nodes[edges[edge_group]]
            left, right = endpoints[:, 0], endpoints[:, 1]
            compatible = (
                (classes[left] == classes[right])
                & (classes[left] >= 0)
                & (classes[left] < self.num_classes)
                & (ids[left] >= 0)
                & (ids[right] >= 0)
            )
            values[edge_group] *= compatible.long()
        graph["edge_label"] = labels

        # Metrics use the same semantic/cluster target partition as the edges.
        # Give each unassigned voxel a distinct ID instead of merging sentinels.
        _, node_ids = torch.unique(
            torch.stack([ids, classes], dim=1), dim=0, return_inverse=True
        )
        invalid = (ids < 0) | (classes < 0) | (classes >= self.num_classes)
        unique_ids = torch.arange(len(ids), device=ids.device) + len(ids)
        node_ids = torch.where(invalid, unique_ids, node_ids)
        graph["node_label"] = TensorBatch(node_ids, clusters.counts)

    def forward(
        self,
        seg_label: TensorBatch,
        clust_label: ClusterLabelBatch,
        ppn_label: TensorBatch,
        vertex_label: TensorBatch | None = None,
        weights: TensorBatch | None = None,
        **result: Any,
    ) -> dict[str, Any]:
        """Evaluate task objectives and combine them with one balancing policy.

        Parameters
        ----------
        seg_label : TensorBatch
            Original voxel semantic truth aligned to the model input.
        clust_label : ClusterLabelBatch
            Original cluster truth on the same voxel domain.
        ppn_label : TensorBatch
            Particle-point proposal labels.
        vertex_label : TensorBatch, optional
            Labels for the optional interaction-vertex proposal branch.
        weights : TensorBatch, optional
            Per-voxel semantic loss weights.
        **result : dict
            Outputs of :class:`UResNetMulti`, including graph diagnostics and,
            when configured, candidate fragment indexes and learned logits.

        Returns
        -------
        dict
            Joint loss and mean task accuracy, separately prefixed task metrics,
            and balancing diagnostics. Empty objectives do not contribute an
            uncertainty regularizer.
        """
        # Supervise preliminary voxel logits and point proposals on original truth
        tasks, terms = self.uresnet_ppn.compute_losses(
            seg_label, ppn_label, vertex_label, clust_label, weights, **result
        )

        # Recover native graph names and label edges without semantic adaptation
        graph = {
            key.removeprefix("graph_spice_"): value
            for key, value in result.items()
            if key.startswith("graph_spice_")
        }
        self.label_graph(seg_label, clust_label, graph)
        edge = self.graph_spice(seg_label, clust_label, **graph)

        # A single agnostic partition has no meaningful per-class metric suffix
        if self.class_agnostic and self.graph_spice.constructor is not None:
            for shape in self.graph_spice.constructor.shapes:
                for metric in ("ari", "purity", "efficiency"):
                    edge.pop(f"{metric}_{shape}", None)
        tasks.append(("graph_spice", edge))
        terms["fragmentation"] = LossTerm(
            edge["loss"], "composite", active=edge["count"] > 0
        )

        # Train the learned classifier on every candidate, including fragments
        # currently predicted outside the set of shapes published downstream.
        if self.fragment_loss is not None:
            fragment = self.fragment_loss(
                seg_label, result["fragment_candidates"], result["fragment_logits"]
            )
            tasks.append(("fragment_semantics", fragment))
            terms["fragment_semantics"] = LossTerm(
                fragment["loss"], "categorical", active=fragment["count"] > 0
            )

        # Preserve the native task sum by default; other policies balance leaves
        if self.loss_balancer.mode == "sum":
            loss = sum(task["loss"] for _, task in tasks)
            metrics = {}
        else:
            loss, metrics = self.loss_balancer(terms)

        output = {"loss": loss, **metrics}
        for prefix, task in tasks:
            output.update({f"{prefix}_{key}": value for key, value in task.items()})
        output["accuracy"] = sum(task["accuracy"] for _, task in tasks) / len(tasks)
        return output
