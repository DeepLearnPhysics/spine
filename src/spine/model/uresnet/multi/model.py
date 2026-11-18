"""Shared UResNet reconstruction with configurable semantic/fragment coupling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch

from spine.data import IndexBatch, TensorBatch
from spine.model.graph_spice.model import GraphSPICEHead
from spine.model.registry import ModelSpec
from spine.model.uresnet.ppn import UResNetPPN

from .loss import UResNetMultiLoss
from .semantics import FragmentSemanticHead, resolve_semantics

__all__ = ["UResNetMulti", "UResNetMultiLoss"]


class UResNetMulti(torch.nn.Module):
    """Combine segmentation, PPN and fragmentation using one UResNet.

    ``hard`` coupling routes graph nodes by preliminary semantic class.
    ``independent`` predicts a class-agnostic partition and pools voxel semantic
    evidence afterward. ``fragment_first`` classifies pooled fragment features
    with a learned head. ``soft`` conditions a class-agnostic edge scorer on
    voxel class probabilities. All modes return semantically coherent fragments.

    Typical configuration:

    .. code-block:: yaml

        model:
          provider: uresnet_multi
          modules:
            uresnet:
              # Shared backbone configuration
            ppn:
              # Point-proposal configuration
            graph_spice:
              # Embedding head, kernel and graph constructor configuration
            semantic_coupling: soft
            fragment_semantics: learned
            semantic_conditioning:
              detach: false
              hidden_channels: 32

    Input voxels must already be deghosted. Discrete graph construction and
    connected components do not carry gradients; the shared features, optional
    semantic conditioning and fragment classifier remain differentiable.
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        ppn: dict[str, Any],
        graph_spice: dict[str, Any],
        vertex: dict[str, Any] | None = None,
        proposal_decoder: dict[str, Any] | None = None,
        semantic_coupling: str = "hard",
        fragment_semantics: str | None = None,
        semantic_conditioning: dict[str, Any] | None = None,
    ) -> None:
        """Initialize one backbone and its task-specific prediction branches.

        Parameters
        ----------
        uresnet : dict
            Backbone and voxel segmentation configuration. Set ``spatial_size``
            for coordinate normalization and leave joint ghost prediction off.
        ppn : dict
            Particle-point proposal configuration.
        graph_spice : dict
            Fragmentation configuration. Its embedder block contains projection
            options only, without a second UResNet configuration.
        vertex : dict, optional
            Additional interaction-vertex proposal configuration.
        proposal_decoder : dict, optional
            Decoder-sharing options for particle and vertex proposals.
        semantic_coupling : {'hard', 'independent', 'fragment_first', 'soft'}
            Dependency between preliminary semantic predictions and fragments.
        fragment_semantics : {'routed', 'pooled', 'learned'}, optional
            Final fragment classification policy. Defaults to routed for hard,
            learned for fragment-first, and pooled for independent/soft coupling.
        semantic_conditioning : dict, optional
            Soft-mode options: ``detach`` controls gradients into semantic logits
            and ``hidden_channels`` sets the correction network width.

        Raises
        ------
        ValueError
            If the policies conflict, fragmentation is disabled when required,
            or a requested fragment class is absent from the semantic head.
        """
        # Initialize the parent class and validate the architectural policies
        super().__init__()
        self.semantic_coupling = semantic_coupling
        self.fragment_semantics = resolve_semantics(
            semantic_coupling, fragment_semantics
        )
        if uresnet.get("ghost", False):
            raise ValueError(
                "uresnet_multi requires deghosted input; use an upstream deghost stage."
            )
        if not isinstance(ppn, dict):
            raise TypeError("uresnet_multi requires a PPN configuration mapping.")

        # Separate the edge-conditioning options from final fragment semantics
        conditioning = (
            {} if semantic_conditioning is None else dict(semantic_conditioning)
        )
        if conditioning and semantic_coupling != "soft":
            raise ValueError("semantic_conditioning is only valid for soft coupling.")
        self.detach_semantics = conditioning.pop("detach", False)
        hidden_channels = conditioning.pop("hidden_channels", 32)
        if not isinstance(self.detach_semantics, bool):
            raise TypeError("semantic_conditioning.detach must be a boolean.")
        if conditioning:
            raise ValueError(
                f"Unknown semantic_conditioning options: {sorted(conditioning)}."
            )

        # The existing semantic/PPN model owns the only UResNet feature pyramid
        self.uresnet_ppn = UResNetPPN(uresnet, ppn, vertex, proposal_decoder)
        backbone = self.uresnet_ppn.uresnet.backbone
        num_classes = uresnet["num_classes"]

        # All non-routed semantic policies require actual predicted fragments
        # during training as well as inference, because they pool over members.
        config = deepcopy(graph_spice)
        config.setdefault("embedder", {})
        config.setdefault("make_clusters", True)
        if self.fragment_semantics != "routed" and not config["make_clusters"]:
            raise ValueError(
                "Pooled or learned fragment semantics require make_clusters=True."
            )

        # Build a backbone-free graph head with optional probability conditioning
        self.graph_spice = GraphSPICEHead(
            config,
            backbone.num_filters,
            backbone.spatial_size,
            backbone.dimension,
            class_agnostic=semantic_coupling != "hard",
            semantic_channels=num_classes if semantic_coupling == "soft" else 0,
            semantic_hidden_channels=hidden_channels,
        )
        if any(shape < 0 or shape >= num_classes for shape in self.graph_spice.shapes):
            raise ValueError(
                "Fragment shapes must be covered by the segmentation head."
            )

        # Final semantics may reuse voxel evidence or introduce a learned head
        self.fragment_head = None
        if self.fragment_semantics != "routed":
            self.fragment_head = FragmentSemanticHead(
                backbone.num_filters, num_classes, self.fragment_semantics
            )

        self.predicts_ppn = True
        self.predicts_vertex = vertex is not None

    def forward(self, data: TensorBatch) -> dict[str, Any]:
        """Run all tasks and enforce one final semantic class per fragment.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + F) Deghosted voxel coordinates and input features.

        Returns
        -------
        dict
            Voxel logits, preliminary/final semantic predictions, proposal
            outputs and namespaced graph predictions. When clustering is enabled,
            canonical fragments use original input indexes. Pooled/learned modes
            also expose all ``fragment_candidates`` and their ``fragment_logits``
            before final output-shape filtering.
        """
        # Evaluate the shared backbone exactly once and retain preliminary logits
        result = self.uresnet_ppn(data)
        segmentation = result["segmentation"]
        seg_pred = TensorBatch(segmentation.torch_tensor().argmax(dim=1), data.counts)
        result["seg_pred_raw"] = seg_pred
        result["seg_pred"] = seg_pred
        features = TensorBatch(
            result["decoder_tensors"][-1].aligned_features(), data.counts
        )

        # Soft conditioning may either jointly train the semantic head through
        # edges or treat its probabilities as fixed evidence for fragmentation.
        probabilities = None
        if self.semantic_coupling == "soft":
            values = torch.softmax(segmentation.torch_tensor(), dim=1)
            if self.detach_semantics:
                values = values.detach()
            probabilities = TensorBatch(values, data.counts)

        # Predict graph edges using the selected coupling policy
        graph = self.graph_spice(data, seg_pred, features, probabilities)
        result.update({f"graph_spice_{key}": value for key, value in graph.items()})
        if "clusts" not in graph:
            return result

        # Restore connected components from the filtered graph domain to input
        # rows before pooling, supervising or publishing canonical fragments.
        index = graph["filter_index"].to_numpy().index
        clusters = graph["clusts"].to_numpy()
        fragments = IndexBatch(
            [index[c] for c in clusters.index_list],
            spans=data.counts,
            counts=clusters.counts,
            single_counts=clusters.single_counts,
            default=index[:0],
        )
        if self.fragment_head is None:
            result["fragment_clusts"] = fragments
            result["fragment_shapes"] = graph["clust_shapes"]
        else:
            self.assign_fragment_semantics(result, features, fragments)

        return result

    def assign_fragment_semantics(
        self,
        result: dict[str, Any],
        features: TensorBatch,
        fragments: IndexBatch,
    ) -> None:
        """Classify candidates, propagate their classes and filter final outputs.

        Parameters
        ----------
        result : dict
            Model results updated in place with coherent semantic products.
        features : TensorBatch
            Shared decoder features aligned to original input rows.
        fragments : IndexBatch
            Candidate fragment partition in the original input domain.

        Notes
        -----
        Every candidate receives one class, which is propagated to its member
        voxels. Candidates assigned a class outside ``graph_spice.shapes`` are
        omitted from canonical fragments but remain available for supervision.
        Voxels omitted by connected-component/orphan handling retain their
        preliminary semantic prediction.
        """
        # Classify every candidate before restricting the published shape set.
        # Otherwise an early wrong prediction could suppress its own supervision.
        assert self.fragment_head is not None
        logits = self.fragment_head(features, result["segmentation"], fragments)
        classes = logits.torch_tensor().argmax(dim=1)
        result["fragment_candidates"] = fragments
        result["fragment_logits"] = logits

        # Give every candidate member its fragment's final class, preserving
        # raw predictions separately for voxel-level supervision and diagnostics.
        final = result["seg_pred_raw"].torch_tensor().clone()
        members = torch.as_tensor(fragments.full_index, device=final.device).long()
        sizes = torch.as_tensor(fragments.single_counts, device=final.device)
        final[members] = torch.repeat_interleave(classes, sizes)
        result["seg_pred"] = TensorBatch(final, features.counts)

        # Select output shapes without changing candidate or loss index domains
        class_array = classes.detach().cpu().numpy()
        keep = np.isin(class_array, self.graph_spice.shapes)
        counts = [
            int(keep[start:stop].sum())
            for start, stop in zip(fragments.edges[:-1], fragments.edges[1:])
        ]
        selected = [c for c, retain in zip(fragments.index_list, keep) if retain]
        result["fragment_clusts"] = IndexBatch(
            selected,
            fragments.spans,
            counts,
            fragments.single_counts[keep],
            default=np.empty(0, dtype=np.int64),
        )
        result["fragment_shapes"] = TensorBatch(class_array[keep], counts)


MODEL_SPEC = ModelSpec("uresnet_multi", UResNetMulti, UResNetMultiLoss)
