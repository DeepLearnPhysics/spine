"""Terminal full-chain providers for materializing GrapPA training inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from spine.constants import DELTA_SHP, MICHL_SHP, SHOWR_SHP, TRACK_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.grappa import GrapPA, GrapPALoss

from ..ops import AggregationOperations
from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult
from .aggregation import CompositeLossStage

__all__ = [
    "GrapPAGraphMaterializationStage",
    "GrapPATargetMaterializationStage",
    "build_fragment_graph_stage",
    "build_fragment_graph_loss",
    "build_particle_graph_stage",
    "build_particle_graph_loss",
]


class GrapPAGraphMaterializationStage(ChainStage):
    """Materialize configured GrapPA graphs without evaluating their GNNs."""

    optional = frozenset({"clust_label", "coord_label", "ppn_points"})
    provides = frozenset()

    def __init__(
        self,
        name: str,
        level: str,
        models: dict[str, GrapPA],
        operations: AggregationOperations,
    ) -> None:
        """Initialize graph materialization for fragment or particle nodes.

        Parameters
        ----------
        name : str
            Stage name.
        level : {"fragment", "particle"}
            Canonical object collection used as graph nodes.
        models : dict of GrapPA
            Native GrapPA model associated with each logical graph path.
        operations : AggregationOperations
            Shared object restriction and model-input preparation helper.
        """
        super().__init__(name)
        self.level = level
        self.models = models
        self.operations = operations
        if level == "fragment":
            self.requires = frozenset(
                {"point_data", "fragment_clusts", "fragment_shapes"}
            )
        elif level == "particle":
            self.requires = frozenset(
                {
                    "point_data",
                    "particle_clusts",
                    "particle_shapes",
                    "particle_primaries",
                }
            )
        else:
            raise ValueError(f"Unknown GrapPA materialization level `{level}`.")

    def _materialize_path(
        self,
        model: GrapPA,
        data: TensorBatch,
        clusts: IndexBatch,
        shapes: TensorBatch,
        accepted_shapes: Sequence[int],
        state: ChainState,
        primaries: IndexBatch | None = None,
    ) -> dict[str, Any]:
        """Build one shape-restricted graph and its encoder features.

        Parameters
        ----------
        model : GrapPA
            Native graph model whose constructors and encoders are reused.
        data : TensorBatch
            Canonical sparse voxel data.
        clusts : IndexBatch
            Candidate graph-node clusters.
        shapes : TensorBatch
            Semantic shape associated with each candidate node.
        accepted_shapes : sequence of int
            Semantic shapes owned by this graph path.
        state : ChainState
            Current chain state containing optional truth and point proposals.
        primaries : IndexBatch, optional
            Primary-member voxels used to associate particle-level points.

        Returns
        -------
        dict
            Native static GrapPA graph products.
        """
        clusts, shapes, _ = self.operations.restrict_clusters(
            clusts,
            shapes,
            accepted_shapes,
        )
        clust_label = state.get("clust_label")
        node_dropout = model.node_dropout
        if (
            node_dropout is not None
            and (node_dropout.group_by is not None or node_dropout.select is not None)
            and clust_label is None
        ):
            raise ValueError(
                "Grouped or selected GrapPA node dropout requires `clust_label` "
                "during graph materialization."
            )
        model_input = self.operations.prepare_grappa_input(
            model,
            data,
            clusts,
            shapes,
            primaries=primaries,
            clust_label=clust_label,
            coord_label=state.get("coord_label"),
            ppn_points=state.get("ppn_points"),
            point_use_primaries=primaries is not None,
        )
        return model.materialize_graph(**model_input)

    def forward(self, state: ChainState) -> StageResult:
        """Materialize every configured graph path and stop before inference.

        Parameters
        ----------
        state : ChainState
            State containing canonical voxel data and graph-node objects.

        Returns
        -------
        StageResult
            Namespaced graph indexes, features and static metadata.
        """
        data = state.require("point_data", self.name).data
        outputs: dict[str, Any] = {}

        if self.level == "fragment":
            clusts = state.require("fragment_clusts", self.name)
            shapes = state.require("fragment_shapes", self.name)
            definitions = {
                "shower": ([SHOWR_SHP, MICHL_SHP, DELTA_SHP], "shower_fragment"),
                "track": ([TRACK_SHP], "track_fragment"),
                "particle": (
                    [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP],
                    "fragment",
                ),
            }
            for path, model in self.models.items():
                accepted_shapes, prefix = definitions[path]
                graph = self._materialize_path(
                    model,
                    data,
                    clusts,
                    shapes,
                    accepted_shapes,
                    state,
                )
                outputs.update(
                    {f"{prefix}_{key}": value for key, value in graph.items()}
                )

        else:
            model = self.models["inter"]
            clusts = state.require("particle_clusts", self.name)
            shapes = state.require("particle_shapes", self.name)
            primaries = state.require("particle_primaries", self.name)
            graph = self._materialize_path(
                model,
                data,
                clusts,
                shapes,
                model.node_type,
                state,
                primaries,
            )
            outputs.update({f"particle_{key}": value for key, value in graph.items()})

        return StageResult(outputs=outputs)


class GrapPATargetMaterializationStage(ChainLossStage):
    """Build namespaced static supervision for one materialized GrapPA graph."""

    def __init__(self, name: str, prefix: str, loss: GrapPALoss) -> None:
        """Initialize a target-materialization adapter.

        Parameters
        ----------
        name : str
            Logical graph path name.
        prefix : str
            Public graph-product prefix removed before native materialization.
        loss : GrapPALoss
            Configured GrapPA objective collection.
        """
        super().__init__(name)
        self.prefix = prefix
        self.loss = loss

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Materialize targets from truth and namespaced graph products.

        Parameters
        ----------
        data : dict
            Driver truth products and full-chain graph outputs.

        Returns
        -------
        dict
            Static target pairs and a neutral loss summary required by the
            full-chain loss aggregator.
        """
        clust_label: ClusterLabelBatch | None = data.get(
            "clust_label_adapt", data.get("clust_label")
        )
        if clust_label is None:
            raise ValueError("GrapPA target materialization requires `clust_label`.")
        native = {
            key.removeprefix(self.prefix): value
            for key, value in data.items()
            if key.startswith(self.prefix)
        }
        result: dict[str, Any] = self.loss.materialize_targets(
            clust_label=clust_label,
            coord_label=data.get("coord_label"),
            graph_label=data.get("graph_label"),
            **native,
        )
        device = next(
            (
                value.device
                for value in native.values()
                if isinstance(value, TensorBatch) and value.device is not None
            ),
            None,
        )
        result.update(
            loss=torch.tensor(0.0, device=device),
            accuracy=1.0,
            num_losses=1,
        )
        return result


def _build_model(key: str, config: Any, owner: Any) -> GrapPA:
    """Construct and register one GrapPA used only for materialization."""
    if not isinstance(config, dict):
        raise ValueError(f"GrapPA materialization requires a `{key}` block.")
    model = GrapPA(config)
    owner.add_module(key, model)
    return model


def build_fragment_graph_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build terminal materialization for fragment-node GrapPA graphs.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Resolved shower, track or joint-particle GrapPA blocks.
    owner : torch.nn.Module
        Full-chain model which owns the configured native modules.

    Returns
    -------
    ChainStage
        Fragment-node graph materialization adapter.
    """
    models = {}
    for path in ("shower", "track", "particle"):
        key = f"grappa_{path}"
        if config.get(key) is not None:
            models[path] = _build_model(key, config[key], owner)
    if not models:
        raise ValueError(
            "Fragment graph materialization requires at least one GrapPA block."
        )
    return GrapPAGraphMaterializationStage(
        name,
        "fragment",
        models,
        AggregationOperations(config.get("predict_points")),
    )


def build_particle_graph_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build terminal materialization for particle-node interaction GrapPA.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Resolved interaction GrapPA block and optional point configuration.
    owner : torch.nn.Module
        Full-chain model which owns the configured native module.

    Returns
    -------
    ChainStage
        Particle-node graph materialization adapter.
    """
    model = _build_model("grappa_inter", config.get("grappa_inter"), owner)
    return GrapPAGraphMaterializationStage(
        name,
        "particle",
        {"inter": model},
        AggregationOperations(config.get("predict_points")),
    )


def build_fragment_graph_loss(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainLossStage | None:
    """Build static-target adapters for fragment-node GrapPA graphs.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Resolved GrapPA model blocks and path-to-loss mapping.
    owner : torch.nn.Module
        Full-chain loss module which owns the native objectives.

    Returns
    -------
    ChainLossStage, optional
        Composite target materializer, or ``None`` without supervision.
    """
    loss_configs = config.get("loss") or {}
    if not isinstance(loss_configs, dict):
        raise TypeError("Fragment graph materialization loss must be a mapping.")
    prefixes = {
        "shower": "shower_fragment_",
        "track": "track_fragment_",
        "particle": "fragment_",
    }
    stages = []
    for path, loss_config in loss_configs.items():
        if not isinstance(loss_config, dict):
            raise TypeError(f"GrapPA `{path}` loss must be a mapping.")
        model_config = config.get(f"grappa_{path}")
        loss = GrapPALoss(loss_config, model_config)
        owner.add_module(f"grappa_{path}_loss", loss)
        stages.append(GrapPATargetMaterializationStage(path, prefixes[path], loss))
    if not stages:
        return None
    return CompositeLossStage(name, stages)


def build_particle_graph_loss(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainLossStage | None:
    """Build static-target materialization for interaction GrapPA.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Resolved interaction GrapPA model and loss blocks.
    owner : torch.nn.Module
        Full-chain loss module which owns the native objective.

    Returns
    -------
    ChainLossStage, optional
        Particle-node target materializer, or ``None`` without supervision.
    """
    loss_config = config.get("loss")
    if loss_config is None:
        return None
    if not isinstance(loss_config, dict):
        raise TypeError("Particle graph materialization loss must be a mapping.")
    loss = GrapPALoss(loss_config, config.get("grappa_inter"))
    owner.add_module("grappa_inter_loss", loss)
    return GrapPATargetMaterializationStage(name, "particle_", loss)


FRAGMENT_GRAPH_SPEC = register_provider(
    ProviderSpec(
        "fragment_graph",
        build_fragment_graph_stage,
        build_fragment_graph_loss,
    )
)
PARTICLE_GRAPH_SPEC = register_provider(
    ProviderSpec(
        "particle_graph",
        build_particle_graph_stage,
        build_particle_graph_loss,
    )
)
