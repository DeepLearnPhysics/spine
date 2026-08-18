"""Semantic-segmentation and point-proposal full-chain provider."""

from __future__ import annotations

from typing import Any

import torch

from spine.constants import GHOST_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.full_chain.label import ClusterLabelAdapter
from spine.model.uresnet import SegmentationLoss, UResNetSegmentation
from spine.model.uresnet.ppn import UResNetPPN, UResNetPPNLoss

from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult


class SegmentationStage(ChainStage):
    """Produce canonical semantic and optional point predictions.

    The stage supports learned UResNet inference and truth-defined semantic
    labels. A combined UResNet-PPN model may additionally select the aligned
    voxel bundle through its ghost prediction and publish point diagnostics.
    """

    requires = frozenset({"point_data"})
    provides = frozenset({"point_data", "seg_pred"})
    replaces = frozenset({"point_data", "clust_label"})

    def __init__(
        self,
        name: str,
        mode: str,
        model: UResNetSegmentation | UResNetPPN | None,
        label_adapter: ClusterLabelAdapter,
    ) -> None:
        """Initialize the semantic provider.

        Parameters
        ----------
        name : str
            Stage name.
        mode : {"uresnet", "label"}
            Source of semantic predictions.
        model : UResNetSegmentation or UResNetPPN, optional
            Native learned model used in ``uresnet`` mode.
        label_adapter : ClusterLabelAdapter
            Utility that aligns structured truth with reconstructed semantics.
        """
        super().__init__(name)
        if mode not in {"uresnet", "label"}:
            raise ValueError("Segmentation mode must be `uresnet` or `label`.")
        self.mode = mode
        self.model = model
        self.label_adapter = label_adapter
        if bool(getattr(model, "predicts_vertex", False)):
            self.provides = self.provides | {"vertex_proposals"}

    def forward(self, state: ChainState) -> StageResult:
        """Run semantic segmentation and adapt aligned cluster labels.

        Parameters
        ----------
        state : ChainState
            State containing voxel data and optional semantic/cluster truth.

        Returns
        -------
        StageResult
            Semantic predictions, native model outputs, and any adapted data
            or cluster-label products.
        """
        point_data = state.require("point_data", self.name)
        data: TensorBatch = point_data.data
        seg_label: TensorBatch | None = state.get("seg_label")
        clust_label: ClusterLabelBatch | None = state.get("clust_label")
        outputs: dict[str, Any] = {}
        products: dict[str, Any] = {}

        # Resolve semantic predictions from truth or the native learned model.
        if self.mode == "label":
            if seg_label is None:
                raise ValueError("Label segmentation requires `seg_label`.")
            seg_pred = TensorBatch(seg_label.values.torch_tensor(), data.counts)
        else:
            if self.model is None:
                raise RuntimeError("The segmentation model was not initialized.")
            model_result = self.model(data)

            # A segmentation backbone may jointly predict a ghost mask. Adapt
            # only row-aligned outputs; sparse intermediate PPN tensors retain
            # their native resolutions.
            if "ghost" in model_result:
                ghost_scores = model_result["ghost"]
                ghost_prediction = torch.argmax(ghost_scores.torch_tensor(), dim=1)
                keep = ghost_prediction == 0
                selected = torch.nonzero(keep, as_tuple=False).flatten()
                adapted = point_data.select(keep)
                adapted_data = adapted.data
                products["point_data"] = adapted
                outputs.update(
                    {
                        "ghost_pred": TensorBatch(ghost_prediction, data.counts),
                    }
                )
                for key in (
                    "ppn_points",
                    "ppn_classify_endpoints",
                    "vertex_points",
                ):
                    if key in model_result:
                        value = model_result[key]
                        model_result[key] = TensorBatch(
                            value.torch_tensor()[selected], adapted_data.counts
                        )
                data = adapted_data

            outputs.update(model_result)
            if "vertex_points" in model_result:
                products["vertex_proposals"] = model_result["vertex_points"]
            segmentation = model_result["segmentation"]
            seg_pred = TensorBatch(
                torch.argmax(segmentation.torch_tensor(), dim=1),
                segmentation.counts,
            )

        products["seg_pred"] = seg_pred
        outputs["seg_pred"] = seg_pred

        # Adapt truth clusters exactly once after the effective voxel set and
        # semantic predictions are known.
        if seg_label is not None and clust_label is not None and self.mode == "uresnet":
            adapted = products.get("point_data", point_data)
            orig_index = adapted.orig_index
            adapted_label = self.label_adapter(
                clust_label,
                seg_label,
                seg_pred,
                orig_index=orig_index,
            )
            products["clust_label"] = adapted_label
            outputs["clust_label_adapt"] = adapted_label

        return StageResult(products, outputs)


class SegmentationLossStage(ChainLossStage):
    """Align truth rows and route semantic or point supervision.

    Cached and on-the-fly deghosting can place truth and logits on different
    row sets. The adapter reconciles those rows before invoking the standalone
    UResNet or UResNet-PPN objective.
    """

    def __init__(
        self,
        name: str,
        loss: SegmentationLoss | UResNetPPNLoss,
    ) -> None:
        """Initialize the native semantic objective.

        Parameters
        ----------
        name : str
            Stage name.
        loss : SegmentationLoss or UResNetPPNLoss
            Standalone semantic or combined semantic/point objective.
        """
        super().__init__(name)
        self.loss = loss

    @staticmethod
    def _align(
        seg_label: TensorBatch,
        segmentation: TensorBatch,
        orig_index: IndexBatch | None,
    ) -> tuple[TensorBatch, TensorBatch]:
        """Align deghosted logits with semantic truth.

        Parameters
        ----------
        seg_label : TensorBatch
            Semantic truth on the original voxel set.
        segmentation : TensorBatch
            Logits on the current effective voxel set.
        orig_index : IndexBatch, optional
            Current-row positions in the original voxel tensor.

        Returns
        -------
        TensorBatch
            Semantic labels aligned with non-ghost predictions.
        TensorBatch
            Corresponding segmentation logits.
        """
        if orig_index is None:
            return seg_label, segmentation
        # Select current rows in the original truth tensor, then remove true
        # ghosts because semantic logits contain only physical voxels.
        selected = torch.zeros(
            seg_label.shape[0],
            dtype=torch.bool,
            device=seg_label.device,
        )
        selected[orig_index.full_index] = True
        aligned_label = seg_label.select(selected)
        keep = aligned_label.values.torch_tensor() < GHOST_SHP
        aligned_label = aligned_label.select(keep)
        aligned_prediction = TensorBatch(
            segmentation.torch_tensor()[keep],
            aligned_label.counts,
        )
        return aligned_label, aligned_prediction

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate semantic and optional point-proposal objectives.

        Parameters
        ----------
        data : dict
            Semantic truth, model logits, and optional PPN products.

        Returns
        -------
        dict
            Native UResNet or UResNet-PPN loss metrics.
        """
        seg_label = data.get("seg_label")
        segmentation = data.get("segmentation")
        if seg_label is None or segmentation is None:
            raise ValueError("Segmentation loss requires labels and logits.")
        seg_label, segmentation = self._align(
            seg_label,
            segmentation,
            data.get("orig_index"),
        )
        # Preserve native standalone loss arguments while overriding products
        # that require full-chain alignment or label adaptation.
        inputs = dict(data)
        inputs.update(
            {
                "seg_label": seg_label,
                "segmentation": segmentation,
                "clust_label": data.get("clust_label_adapt", data.get("clust_label")),
            }
        )
        return self.loss(**inputs)


def build_segmentation_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build the selected segmentation backbone and label adapter.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Semantic mode, native model block, and label-adapter options.
    owner : torch.nn.Module
        Full-chain model that owns the native network.

    Returns
    -------
    ChainStage
        Configured semantic stage.
    """
    mode = config.get("mode")
    if not isinstance(mode, str):
        raise ValueError("Segmentation requires a string `mode`.")

    point_proposal = config.get("point_proposal")
    # Learned mode accepts either standalone UResNet or the combined PPN model.
    model = None
    if mode == "uresnet":
        uresnet = config.get("uresnet")
        uresnet_ppn = config.get("uresnet_ppn")
        if (uresnet is None) == (uresnet_ppn is None):
            raise ValueError("Provide exactly one of `uresnet` and `uresnet_ppn`.")
        if point_proposal == "ppn" and uresnet_ppn is None:
            raise ValueError("PPN point proposal requires `uresnet_ppn`.")
        if uresnet_ppn is not None:
            if not isinstance(uresnet_ppn, dict):
                raise TypeError("`uresnet_ppn` configuration must be a mapping.")
            model = UResNetPPN(**uresnet_ppn)
            owner.add_module("uresnet_ppn", model)
        else:
            if not isinstance(uresnet, dict):
                raise TypeError("`uresnet` configuration must be a mapping.")
            model = UResNetSegmentation(uresnet)
            owner.add_module("uresnet", model)

    # Label adaptation is independent of the selected semantic implementation.
    adapter_config = config.get("adapt_labels") or {}
    if not isinstance(adapter_config, dict):
        raise TypeError("`adapt_labels` configuration must be a mapping.")
    return SegmentationStage(name, mode, model, ClusterLabelAdapter(**adapter_config))


def build_segmentation_loss(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainLossStage | None:
    """Build semantic or combined semantic/PPN supervision.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Resolved model and loss blocks.
    owner : torch.nn.Module
        Full-chain loss module that owns the native objective.

    Returns
    -------
    ChainLossStage or None
        Semantic loss adapter, or ``None`` without supervision.
    """
    loss_config = config.get("loss")
    if loss_config is None:
        return None
    if not isinstance(loss_config, dict):
        raise TypeError("Segmentation loss configuration must be a mapping.")

    # Model-block presence selects the matching standalone loss constructor.
    ppn_config = config.get("uresnet_ppn")
    if ppn_config is not None:
        if not isinstance(ppn_config, dict):
            raise TypeError("`uresnet_ppn` configuration must be a mapping.")
        loss = UResNetPPNLoss(**ppn_config, **loss_config)
        owner.add_module("uresnet_ppn_loss", loss)
    else:
        model_config = config.get("uresnet")
        if not isinstance(model_config, dict):
            raise ValueError("UResNet loss requires a `uresnet` model block.")
        loss = SegmentationLoss(model_config, loss_config)
        owner.add_module("uresnet_loss", loss)
    return SegmentationLossStage(name, loss)


PROVIDER_SPEC = register_provider(
    ProviderSpec("segmentation", build_segmentation_stage, build_segmentation_loss)
)
