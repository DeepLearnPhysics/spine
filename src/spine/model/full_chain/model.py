"""Ordered, provider-driven end-to-end reconstruction chain."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from spine.data import ClusterLabelBatch, IndexBatch, RunInfo, TensorBatch

from ..registry import ModelSpec
from .config import StageConfig, build_chain_plan
from .point import PointBatch
from .registry import provider_spec
from .stage import ChainLossStage, ChainStage
from .state import ChainState, StageResult

__all__ = ["FullChain", "FullChainLoss", "process_chain_config", "MODEL_SPEC"]


class FullChain(torch.nn.Module):
    """Execute an ordered plan of reconstruction capability providers.

    The orchestrator knows only about canonical products and stage contracts.
    Native models retain their standalone interfaces; provider adapters own
    all translation into products such as ``seg_pred``, ``fragment_clusts``
    and ``particle_clusts``. A new implementation can consequently replace or
    jointly provide capabilities without modifying this class.

    Both the historical ``chain`` mode matrix and the native ordered ``stages``
    syntax are accepted. The former is translated once during construction.
    """

    _input_products = {
        "data",
        "sources",
        "seg_label",
        "clust_label",
        "orig_index",
        "coord_label",
        "energy_label",
        "meta",
        "run_info",
        "point_data",
    }

    def __init__(self, chain: dict[str, Any], **modules: Any) -> None:
        """Initialize and validate the configured execution plan.

        Parameters
        ----------
        chain : dict
            Historical mode matrix or native ordered-stage configuration.
        **modules : dict
            Named native model and provider configuration blocks.
        """
        super().__init__()
        self.plan = build_chain_plan(chain, modules)
        self.stages: list[ChainStage] = []

        # Providers register their native torch modules directly on this owner.
        # Existing attribute names are retained for checkpoint compatibility.
        available = set(self._input_products)
        for stage_config in self.plan:
            spec = provider_spec(stage_config.provider)
            stage = spec.stage(stage_config.name, stage_config.config, self)
            available = stage.validate(available)
            self.stages.append(stage)

    def forward(
        self,
        data: TensorBatch,
        sources: TensorBatch | None = None,
        seg_label: TensorBatch | None = None,
        clust_label: ClusterLabelBatch | None = None,
        orig_index: IndexBatch | None = None,
        coord_label: TensorBatch | None = None,
        energy_label: TensorBatch | None = None,
        meta: Sequence[Any] | None = None,
        run_info: Sequence[RunInfo] | None = None,
        **products: Any,
    ) -> dict[str, Any]:
        """Run all stages and return their stable public outputs.

        Parameters
        ----------
        data : TensorBatch
            Canonical sparse voxel input.
        sources : TensorBatch, optional
            Voxel-aligned detector-source identifiers.
        seg_label : TensorBatch, optional
            Voxel-level semantic truth.
        clust_label : ClusterLabelBatch, optional
            Compact structured particle and cluster truth.
        orig_index : IndexBatch, optional
            Cached mapping from current rows to an original voxel tensor.
        coord_label : TensorBatch, optional
            Particle start/end point truth.
        energy_label : TensorBatch, optional
            Voxel-level deposited-energy truth.
        meta : sequence, optional
            Event image metadata used by detector calibration.
        run_info : sequence of RunInfo, optional
            Event identifiers used by time-dependent calibration.
        **products : object, optional
            Additional products consumed by externally registered providers.

        Returns
        -------
        dict
            Public native diagnostics and canonical reconstruction products.
        """
        point_data = PointBatch.from_input(data, sources, orig_index)
        state = ChainState(
            data=data,
            sources=sources,
            seg_label=seg_label,
            clust_label=clust_label,
            orig_index=orig_index,
            coord_label=coord_label,
            energy_label=energy_label,
            meta=meta,
            run_info=run_info,
            point_data=point_data,
            **products,
        )

        # The chain loop is intentionally ignorant of concrete model types.
        for stage in self.stages:
            result = stage(state)
            state.publish(stage.name, result, stage.replaces)

        # Delay public aliases until every row-changing stage has run. This
        # guarantees data_adapt, data_calib, sources, and orig_index all expose
        # the final common row domain, independent of configured stage order.
        point_outputs = state.require("point_data").public_outputs()
        if point_outputs:
            state.publish(
                "full_chain",
                StageResult(outputs=point_outputs),
            )

        return state.outputs


class FullChainLoss(torch.nn.Module):
    """Aggregate objectives owned by configured full-chain providers.

    The loss follows the same normalized plan as :class:`FullChain`, but each
    provider builds a lightweight adapter around its standalone objective.
    Component metrics are namespaced by stage while the total loss and mean
    accuracy retain the model-manager interface.
    """

    def __init__(self, chain: dict[str, Any], **modules: Any) -> None:
        """Build loss adapters from the normalized model plan.

        Parameters
        ----------
        chain : dict
            Historical mode matrix or native ordered-stage configuration.
        **modules : dict
            Named model and loss blocks. Model blocks provide context needed
            to initialize their corresponding objectives.
        """
        super().__init__()
        self.plan = build_chain_plan(chain, modules, require_losses=True)
        self.stages: list[ChainLossStage] = []

        # Loss builders receive their corresponding model configuration as
        # context, but instantiate only the provider-owned objective modules.
        for stage_config in self.plan:
            spec = provider_spec(stage_config.provider)
            if spec.loss is None:
                continue
            config = dict(stage_config.config)
            config["loss"] = stage_config.loss_config
            stage = spec.loss(stage_config.name, config, self)
            if stage is not None:
                self.stages.append(stage)

    def forward(self, **data: Any) -> dict[str, Any]:
        """Evaluate configured objectives and combine summary metrics.

        Parameters
        ----------
        **data : object
            Driver loss inputs together with public outputs from
            :class:`FullChain`.

        Returns
        -------
        dict
            Total loss, mean accuracy, objective count, and namespaced
            component diagnostics.

        Raises
        ------
        ValueError
            If a provider reports an empty objective collection.
        """
        result: dict[str, Any] = {"loss": 0.0, "accuracy": 1.0, "num_losses": 0}

        # Accumulate provider summaries using objective counts as weights.
        for stage in self.stages:
            stage_result = stage(data)
            count = int(stage_result.get("num_losses", 1))
            if count < 1:
                raise ValueError(f"Loss stage `{stage.name}` reported no objectives.")

            previous = result["num_losses"]
            result["loss"] = result["loss"] + stage_result["loss"]
            result["accuracy"] = (
                result["accuracy"] * previous
                + float(stage_result.get("accuracy", 1.0)) * count
            ) / (previous + count)
            result["num_losses"] = previous + count

            # Keep component diagnostics unambiguous across interchangeable
            # providers while retaining the chain-wide summary keys above.
            for key, value in stage_result.items():
                if key not in {"loss", "accuracy", "num_losses"}:
                    result[f"{stage.name}_{key}"] = value
        return result


def process_chain_config(
    owner: Any,
    dump_config: bool = False,
    **parameters: Any,
) -> list[StageConfig]:
    """Compatibility wrapper for callers of the historical config helper.

    This helper validates the old mode matrix through the same translator used
    by :class:`FullChain` and mirrors its keys onto ``owner``. New code should
    call :func:`build_chain_plan` directly.

    Parameters
    ----------
    owner : object
        Object receiving historical mode attributes.
    dump_config : bool, default False
        Retained for API compatibility; configuration logging is now handled by
        the surrounding model manager.
    **parameters : object
        Historical chain mode matrix.

    Returns
    -------
    list of StageConfig
        Normalized provider execution plan.

    Notes
    -----
    ``dump_config`` is intentionally ignored. Configuration logging is owned
    by the model manager in the provider-based implementation.
    """
    del dump_config
    plan = build_chain_plan(parameters, {})
    for name, value in parameters.items():
        setattr(owner, name, value)
    return plan


MODEL_SPEC = ModelSpec("full_chain", FullChain, FullChainLoss)
