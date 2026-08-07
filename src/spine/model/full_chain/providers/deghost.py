"""Deghosting provider for the full reconstruction chain."""

from __future__ import annotations

from typing import Any

import torch

from spine.constants import GHOST_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.uresnet import SegmentationLoss, UResNetSegmentation
from spine.utils.ghost import ChargeRescaler

from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult


class DeghostStage(ChainStage):
    """Remove predicted or truth-labeled ghost voxels from canonical data.

    The stage selects the aligned voxel bundle once, applying the same mask to
    active data, input charge, calibrated data, sources, and original indexes.
    """

    requires = frozenset({"point_data"})
    provides = frozenset({"point_data"})
    replaces = frozenset({"point_data"})

    def __init__(
        self,
        name: str,
        mode: str | None,
        model: UResNetSegmentation | None,
        charge_rescaling: str | None,
    ) -> None:
        """Initialize the deghosting implementation.

        Parameters
        ----------
        name : str
            Stage name.
        mode : {"uresnet", "label"}
            Source of the binary ghost decision.
        model : UResNetSegmentation, optional
            Binary semantic model used in ``uresnet`` mode.
        charge_rescaling : {"collection", "average", "label"}, optional
            Charge correction applied after ghost removal.
        """
        super().__init__(name)
        if mode not in {"uresnet", "label"}:
            raise ValueError("Deghosting mode must be `uresnet` or `label`.")
        if charge_rescaling not in {None, "collection", "average", "label"}:
            raise ValueError(f"Unknown charge-rescaling mode `{charge_rescaling}`.")
        if charge_rescaling == "label" and mode != "label":
            raise ValueError("Label charge rescaling requires label deghosting.")
        self.mode = mode
        self.model = model
        self.charge_rescaling = charge_rescaling
        self.charge_rescaler = (
            ChargeRescaler(collection_only=charge_rescaling == "collection")
            if charge_rescaling in {"collection", "average"}
            else None
        )

    def forward(self, state: ChainState) -> StageResult:
        """Deghost the current voxel tensor and aligned source information.

        Parameters
        ----------
        state : ChainState
            State containing voxel data and optional truth/source products.

        Returns
        -------
        StageResult
            Adapted data, original-row indexes, ghost predictions, and any
            aligned source products.
        """
        point_data = state.require("point_data", self.name)
        data: TensorBatch = point_data.data
        sources: TensorBatch | None = point_data.sources
        seg_label: TensorBatch | None = state.get("seg_label")
        clust_label: ClusterLabelBatch | None = state.get("clust_label")

        # Build the non-ghost mask from either model predictions or truth labels.
        outputs: dict[str, Any] = {}
        if self.mode == "uresnet":
            if self.model is None:
                raise RuntimeError("The deghosting model was not initialized.")
            model_result = self.model(data)
            ghost_scores = model_result["segmentation"]
            ghost_prediction = torch.argmax(ghost_scores.torch_tensor(), dim=1)
            outputs["ghost"] = ghost_scores
        else:
            if seg_label is None:
                raise ValueError("Label deghosting requires `seg_label`.")
            ghost_prediction = (seg_label.values.torch_tensor() == GHOST_SHP).long()

        # Apply one shared row selection and retain its mapping to original
        # input positions for downstream truth and loss alignment.
        keep = ghost_prediction == 0
        adapted = point_data.select(keep)
        ghost_pred = TensorBatch(ghost_prediction, data.counts)

        # Optionally replace charge with rescaled reconstruction or truth values.
        if self.charge_rescaler is not None:
            values = self.charge_rescaler(adapted.data)
            adapted = adapted.with_charge(values)
        elif self.charge_rescaling == "label":
            if clust_label is None:
                raise ValueError("Label charge rescaling requires `clust_label`.")
            adapted = adapted.with_charge(clust_label.values.torch_tensor()[keep])

        products: dict[str, Any] = {"point_data": adapted}
        outputs.update(
            {
                "ghost_pred": ghost_pred,
            }
        )

        # Preserve independently useful truth and source alignment products.
        if seg_label is not None:
            truth_keep = seg_label.values.torch_tensor() != GHOST_SHP
            truth_index = torch.nonzero(truth_keep, as_tuple=False).flatten()
            truth_counts = seg_label.select(truth_keep).counts
            outputs["orig_index_label"] = IndexBatch(
                truth_index,
                spans=seg_label.counts,
                counts=truth_counts,
            )
        if sources is not None:
            if seg_label is not None:
                truth_keep = seg_label.values.torch_tensor() != GHOST_SHP
                truth_counts = seg_label.select(truth_keep).counts
                outputs["sources_label"] = TensorBatch(
                    sources.torch_tensor()[truth_keep], truth_counts
                )

        return StageResult(products, outputs)


class DeghostLossStage(ChainLossStage):
    """Convert semantic truth into binary labels for deghost supervision.

    The underlying segmentation objective remains unaware of full-chain shape
    conventions and receives zero for physical voxels and one for ghosts.
    """

    def __init__(self, name: str, loss: SegmentationLoss) -> None:
        """Initialize the native segmentation objective.

        Parameters
        ----------
        name : str
            Stage name.
        loss : SegmentationLoss
            Binary segmentation objective.
        """
        super().__init__(name)
        self.loss = loss

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the deghosting objective.

        Parameters
        ----------
        data : dict
            Semantic truth and predicted ghost logits.

        Returns
        -------
        dict
            Native segmentation loss metrics.
        """
        seg_label = data.get("seg_label")
        ghost = data.get("ghost")
        if seg_label is None or ghost is None:
            raise ValueError("Deghosting loss requires `seg_label` and `ghost`.")
        labels = TensorBatch(
            (seg_label.values.torch_tensor() == GHOST_SHP).long(),
            seg_label.counts,
        )
        return self.loss(seg_label=labels, segmentation=ghost)


def build_deghost_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build and register the configured deghosting model.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Deghosting mode, charge-rescaling mode, and optional UResNet block.
    owner : torch.nn.Module
        Full-chain model that owns native trainable modules.

    Returns
    -------
    ChainStage
        Configured deghosting adapter.
    """
    mode = config.get("mode")
    model = None
    if mode == "uresnet":
        model_config = config.get("uresnet_deghost")
        if not isinstance(model_config, dict):
            raise ValueError("UResNet deghosting requires `uresnet_deghost` config.")
        model = UResNetSegmentation(model_config)
        owner.add_module("uresnet_deghost", model)
    return DeghostStage(name, mode, model, config.get("charge_rescaling"))


def build_deghost_loss(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainLossStage | None:
    """Build deghost supervision when a loss block is configured.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Provider configuration and resolved loss block.
    owner : torch.nn.Module
        Full-chain loss module that owns the native objective.

    Returns
    -------
    ChainLossStage or None
        Deghosting loss adapter, or ``None`` when supervision is disabled.
    """
    model_config = config.get("uresnet_deghost")
    loss_config = config.get("loss")
    if model_config is None or loss_config is None:
        return None
    if not isinstance(model_config, dict) or not isinstance(loss_config, dict):
        raise TypeError("Deghosting model and loss blocks must be mappings.")
    loss = SegmentationLoss(model_config, loss_config)
    owner.add_module("deghost_loss", loss)
    return DeghostLossStage(name, loss)


PROVIDER_SPEC = register_provider(
    ProviderSpec("deghost", build_deghost_stage, build_deghost_loss)
)
