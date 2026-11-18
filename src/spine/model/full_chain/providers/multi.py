"""Combined semantic, point-proposal and fragmentation chain provider."""

from __future__ import annotations

from typing import Any

import torch

from spine.data import TensorBatch
from spine.model.full_chain.label import ClusterLabelAdapter
from spine.model.uresnet.multi import UResNetMulti, UResNetMultiLoss

from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult


class MultiStage(ChainStage):
    """Publish semantic, proposal and fragment products from one shared model.

    The provider replaces the separate segmentation and fragmentation stages.
    Final coherent voxel classes and fragment classes are published together,
    before any downstream label adaptation or graph materialization.
    """

    requires = frozenset({"point_data"})
    provides = frozenset(
        {"seg_pred", "ppn_points", "fragment_clusts", "fragment_shapes"}
    )
    replaces = frozenset({"clust_label"})

    def __init__(
        self,
        name: str,
        model: UResNetMulti,
        label_adapter: ClusterLabelAdapter | None,
    ) -> None:
        """Initialize the combined reconstruction stage.

        Parameters
        ----------
        name : str
            Unique stage name in the ordered full-chain plan.
        model : UResNetMulti
            Shared model configured to return canonical fragments.
        label_adapter : ClusterLabelAdapter, optional
            Adapter for downstream reconstructed-domain objectives. Original
            truth remains available to the combined model's own loss.
        """
        # Store the model and declare the optional vertex product
        super().__init__(name)
        self.model = model
        self.label_adapter = label_adapter
        if model.predicts_vertex:
            self.provides = self.provides | {"vertex_proposals"}

    def forward(self, state: ChainState) -> StageResult:
        """Run all shared tasks and publish their canonical products.

        Parameters
        ----------
        state : ChainState
            Current aligned point data and optional original truth products.

        Returns
        -------
        StageResult
            Final semantic and fragment products, point proposals and native
            diagnostics. Adapted cluster truth is included when configured.
        """
        # Evaluate the combined model once on the current point-data domain
        points = state.require("point_data", self.name)
        outputs = self.model(points.data)

        # Publish final semantics together with the partition that defines them
        products = {
            key: outputs[key] for key in self.provides if key != "vertex_proposals"
        }
        if self.model.predicts_vertex:
            products["vertex_proposals"] = outputs["vertex_points"]

        # Only downstream objectives consume prediction-adapted cluster truth
        seg_label = state.get("seg_label")
        clust_label = state.get("clust_label")
        if (
            self.label_adapter is not None
            and seg_label is not None
            and clust_label is not None
        ):
            adapted = self.label_adapter(
                clust_label,
                seg_label,
                outputs["seg_pred"],
                orig_index=points.orig_index,
            )
            products["clust_label"] = adapted
            outputs["clust_label_adapt"] = adapted

        return StageResult(products, outputs)


class MultiLossStage(ChainLossStage):
    """Align original supervision to the shared model's selected voxel domain.

    Upstream deghosting can change voxel membership while preserving the driver
    truth products. This adapter applies the original-row mapping to voxel
    labels, retaining coordinate schemas and leaving point-level labels intact.
    """

    def __init__(self, name: str, loss: UResNetMultiLoss) -> None:
        """Initialize the combined objective adapter.

        Parameters
        ----------
        name : str
            Name of the corresponding reconstruction stage.
        loss : UResNetMultiLoss
            Joint objective with the same coupling policy as the network.
        """
        super().__init__(name)
        self.loss = loss

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Align original voxel labels and evaluate the native joint objective.

        Parameters
        ----------
        data : dict
            Driver truth and public model outputs. ``orig_index``, if present,
            maps current voxels to their original driver-input rows.

        Returns
        -------
        dict
            Native joint loss, task metrics and balancing diagnostics.
        """
        # Never replace original truth with the downstream adapted labels
        inputs = dict(data)
        index = data.get("orig_index")
        if index is not None:
            selected = index.full_index
            counts = data["segmentation"].counts

            # Preserve physical column metadata as well as the logical schema
            for key in ("seg_label", "weights"):
                if inputs.get(key) is not None:
                    value = inputs[key]
                    inputs[key] = TensorBatch(
                        value.torch_tensor()[selected],
                        counts,
                        has_batch_col=value.has_batch_col,
                        coord_cols=value.coord_cols,
                        schema=value.schema,
                        meta=value.meta,
                    )

            # Structured cluster labels maintain their own particle references
            if inputs.get("clust_label") is not None:
                inputs["clust_label"] = inputs["clust_label"].select(selected, counts)

        return self.loss(**inputs)


def build_multi_stage(
    name: str, config: dict[str, Any], owner: torch.nn.Module
) -> MultiStage:
    """Build a shared reconstruction provider with canonical fragment outputs.

    Parameters
    ----------
    name : str
        Stage instance name.
    config : dict
        Native ``uresnet_multi`` block and optional ``adapt_labels`` policy.
    owner : torch.nn.Module
        Full-chain network that owns the registered shared model parameters.

    Returns
    -------
    MultiStage
        Provider replacing separate segmentation and fragmentation stages.

    Raises
    ------
    ValueError
        If the model is configured not to construct fragments.
    """
    # Instantiate the native model and enforce the provider's product contract
    model = UResNetMulti(**config["uresnet_multi"])
    if not model.graph_spice.make_clusters:
        raise ValueError(
            "The uresnet_multi chain provider requires make_clusters=True."
        )
    owner.add_module("uresnet_multi", model)

    # Label adaptation belongs after final semantic assignment, regardless of
    # whether those classes came from routing, pooling or a learned classifier.
    adapt = config.get("adapt_labels", True)
    if adapt is False:
        adapter = None
    elif adapt is True or adapt is None:
        adapter = ClusterLabelAdapter()
    elif isinstance(adapt, dict):
        adapter = ClusterLabelAdapter(**adapt)
    else:
        raise TypeError("adapt_labels must be a boolean or mapping.")

    return MultiStage(name, model, adapter)


def build_multi_loss(
    name: str, config: dict[str, Any], owner: torch.nn.Module
) -> MultiLossStage | None:
    """Build the loss adapter associated with a combined reconstruction stage.

    Parameters
    ----------
    name : str
        Name of the corresponding network stage.
    config : dict
        Resolved native model configuration and optional ``loss`` block.
    owner : torch.nn.Module
        Full-chain objective that owns the registered native loss parameters.

    Returns
    -------
    MultiLossStage or None
        Joint objective adapter, or ``None`` when no supervision is configured.
    """
    # Inference-only stages do not need a native objective
    if config.get("loss") is None:
        return None

    # Match the network's coupling and classification policies exactly
    loss = UResNetMultiLoss(**config["uresnet_multi"], **config["loss"])
    owner.add_module("uresnet_multi_loss", loss)
    return MultiLossStage(name, loss)


PROVIDER_SPEC = register_provider(
    ProviderSpec("uresnet_multi", build_multi_stage, build_multi_loss)
)
