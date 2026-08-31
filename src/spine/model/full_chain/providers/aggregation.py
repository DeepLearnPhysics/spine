"""Particle and interaction aggregation providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from spine.constants import DELTA_SHP, MICHL_SHP, SHOWR_SHP, TRACK_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.grappa import GrapPA, GrapPALoss

from ..ops import AggregationOperations
from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult


class ParticleAggregationStage(ChainStage):
    """Aggregate semantic fragments into complete particle candidates.

    Shower and track fragments may be processed by separate implementations,
    or all supported shapes may be processed by one joint path. Each path
    produces the same canonical particle, shape, and primary products.
    """

    requires = frozenset({"point_data", "fragment_clusts", "fragment_shapes"})
    optional = frozenset({"clust_label", "coord_label", "ppn_points"})
    provides = frozenset({"particle_clusts", "particle_shapes", "particle_primaries"})

    def __init__(
        self,
        name: str,
        modes: dict[str, str | None],
        models: dict[str, GrapPA],
        operations: AggregationOperations,
    ) -> None:
        """Initialize particle aggregation modes and native models.

        Parameters
        ----------
        name : str
            Stage name.
        modes : dict
            Aggregation mode for ``shower``, ``track``, and ``particle`` paths.
        models : dict
            Native GrapPA model for each learned path.
        operations : AggregationOperations
            Shared input-preparation and group-construction helper.
        """
        super().__init__(name)
        self.modes = modes
        self.models = models
        self.operations = operations

    @staticmethod
    def _empty(fragments: IndexBatch) -> tuple[IndexBatch, TensorBatch, IndexBatch]:
        """Create empty particle products using fragment metadata.

        Parameters
        ----------
        fragments : IndexBatch
            Fragment collection defining batch spans.

        Returns
        -------
        IndexBatch
            Empty particle clusters.
        TensorBatch
            Empty particle shapes.
        IndexBatch
            Empty particle-primary clusters.
        """
        counts = np.zeros(fragments.batch_size, dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        particles = IndexBatch([], fragments.spans, counts, [], default=empty)
        shapes = TensorBatch(empty, counts)
        primaries = IndexBatch([], fragments.spans, counts, [], default=empty)
        return particles, shapes, primaries

    def _run_path(
        self,
        path: str,
        accepted_shapes: Sequence[int],
        use_primary: bool,
        state: ChainState,
    ) -> tuple[IndexBatch, TensorBatch, IndexBatch, np.ndarray | None, dict[str, Any]]:
        """Run one shower, track, or joint-particle aggregation path.

        Parameters
        ----------
        path : {"shower", "track", "particle"}
            Logical aggregation path.
        accepted_shapes : sequence of int
            Semantic shapes owned by the path.
        use_primary : bool
            Whether group shape and retained voxels use primary predictions.
        state : ChainState
            Current chain state.

        Returns
        -------
        IndexBatch
            Grouped particle clusters.
        TensorBatch
            Particle shapes.
        IndexBatch
            Primary-fragment voxel indexes.
        numpy.ndarray or None
            Positions of path fragments in the full fragment collection.
        dict
            Native model outputs.
        """
        fragments: IndexBatch = state.require("fragment_clusts", self.name)
        fragment_shapes: TensorBatch = state.require("fragment_shapes", self.name)
        mode = self.modes[path]

        # Learned paths delegate native input preparation and grouping.
        if mode == "grappa":
            model = self.models[path]
            return self.operations.run_grappa(
                model,
                state.require("point_data", self.name).data,
                fragments,
                fragment_shapes,
                accepted_shapes,
                clust_label=state.get("clust_label"),
                coord_label=state.get("coord_label"),
                ppn_points=state.get("ppn_points"),
                aggregate_shapes=True,
                shape_use_primary=use_primary,
                retain_primaries=use_primary,
            )

        # Truth paths reduce structured labels over the same fragment inputs.
        if mode == "label":
            clust_label: ClusterLabelBatch | None = state.get("clust_label")
            if clust_label is None:
                raise ValueError("Label particle aggregation requires `clust_label`.")
            groups = self.operations.group_labels(
                clust_label,
                fragments,
                fragment_shapes,
                accepted_shapes,
                aggregate_shapes=True,
                shape_use_primary=use_primary,
                retain_primaries=use_primary,
            )
            return groups[0], groups[1], groups[2], groups[3], {}

        # Skip mode promotes each accepted fragment to one particle.
        if mode == "skip":
            groups, shapes, shape_index = self.operations.restrict_clusters(
                fragments,
                fragment_shapes,
                accepted_shapes,
            )
            return groups, shapes, groups, shape_index, {}

        raise RuntimeError(f"Particle aggregation path `{path}` is disabled.")

    def forward(self, state: ChainState) -> StageResult:
        """Run configured paths and merge their particle lists.

        Parameters
        ----------
        state : ChainState
            State containing canonical voxel and fragment products.

        Returns
        -------
        StageResult
            Particle clusters, shapes, primaries, and namespaced native model
            diagnostics.
        """
        data: TensorBatch = state.require("point_data", self.name).data
        fragments: IndexBatch = state.require("fragment_clusts", self.name)
        particles, particle_shapes, particle_primaries = self._empty(fragments)
        outputs: dict[str, Any] = {}

        # Associate each logical path with its semantic ownership and primary
        # behavior before entering the common execution loop.
        definitions = {
            "shower": ([SHOWR_SHP, MICHL_SHP, DELTA_SHP], True),
            "track": ([TRACK_SHP], False),
            "particle": ([SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP], True),
        }
        # Separate learned shower/track paths also expose one fragment-aligned
        # diagnostic view consumed by historical reconstruction builders.
        merged: dict[str, Any] = {}
        if self.modes["shower"] == "grappa" or self.modes["track"] == "grappa":
            num_fragments = len(fragments.index_list)
            kwargs = {"dtype": data.dtype, "device": data.device}
            merged = {
                "start_points": torch.full((num_fragments, 3), -torch.inf, **kwargs),
                "end_points": torch.full((num_fragments, 3), -torch.inf, **kwargs),
                "node_pred": torch.full((num_fragments, 2), -torch.inf, **kwargs),
                "group_pred": -np.ones(num_fragments, dtype=np.int64),
            }

        # Separate shower/track paths and the joint-particle path are mutually
        # exclusive by construction, but share the same output contract.
        for path, (accepted_shapes, use_primary) in definitions.items():
            if self.modes[path] is None:
                continue
            groups, shapes, primaries, shape_index, native = self._run_path(
                path,
                accepted_shapes,
                use_primary,
                state,
            )
            prefix = f"{path}_fragment" if path != "particle" else "fragment"
            outputs.update({f"{prefix}_{key}": value for key, value in native.items()})

            # Merge paths in batch order into the canonical particle products.
            particles = particles.merge(groups)
            particle_shapes = particle_shapes.merge(shapes)
            particle_primaries = particle_primaries.merge(primaries)

            if merged and shape_index is not None:
                for key, target in merged.items():
                    source_key = f"{prefix}_{key}"
                    if source_key not in outputs:
                        continue
                    source = outputs[source_key]
                    if isinstance(target, torch.Tensor):
                        index = torch.as_tensor(
                            shape_index,
                            dtype=torch.long,
                            device=target.device,
                        )
                        target[index] = source.torch_tensor()
                        continue

                    values = source.to_numpy().tensor
                    if key == "group_pred":
                        values = values + np.max(target, initial=-1) + 1
                    target[shape_index] = values

        # Preserve the historical merged fragment diagnostics consumed by the
        # reconstruction builders after separate shower and track aggregation.
        for key, value in merged.items():
            coord_cols = np.arange(3) if key.endswith("points") else None
            batch = TensorBatch(value, fragments.counts, coord_cols=coord_cols)
            if key == "group_pred":
                offset = 0
                normalized = batch.numpy_tensor()
                for batch_id, values in enumerate(batch.split()):
                    lower, upper = batch.edges[batch_id : batch_id + 2]
                    inverse = np.unique(values, return_inverse=True)[1]
                    normalized[lower:upper] = offset + inverse
                    offset += len(np.unique(inverse))
            outputs[f"fragment_{key}"] = batch

        products = {
            "particle_clusts": particles,
            "particle_shapes": particle_shapes,
            "particle_primaries": particle_primaries,
        }
        outputs.update(products)
        return StageResult(products, outputs)


class InteractionAggregationStage(ChainStage):
    """Aggregate reconstructed particles into interaction candidates.

    Interaction GrapPA may jointly predict particle attributes. Individual
    attributes delegated to the image provider are suppressed here to keep
    public output ownership unambiguous.
    """

    requires = frozenset(
        {"point_data", "particle_clusts", "particle_shapes", "particle_primaries"}
    )
    optional = frozenset({"clust_label", "coord_label", "ppn_points"})
    provides = frozenset({"interaction_clusts"})

    def __init__(
        self,
        name: str,
        mode: str,
        model: GrapPA | None,
        operations: AggregationOperations,
        task_modes: dict[str, str | None] | None = None,
    ) -> None:
        """Initialize interaction aggregation.

        Parameters
        ----------
        name : str
            Stage name.
        mode : {"grappa", "label"}
            Learned or truth-defined interaction grouping.
        model : GrapPA, optional
            Native interaction graph model.
        operations : AggregationOperations
            Shared graph input and grouping helper.
        task_modes : dict, optional
            Provider ownership for particle-level node tasks.
        """
        super().__init__(name)
        if mode not in {"grappa", "label"}:
            raise ValueError(
                "Interaction aggregation mode must be `grappa` or `label`."
            )
        self.mode = mode
        self.model = model
        self.operations = operations
        self.task_modes = task_modes or {}

        # A named vertex head exposes canonical particle-level proposals for a
        # later interaction reducer. Keep the native diagnostics unchanged.
        self.predicts_vertex = bool(
            model is not None
            and "node_vertex_pred" in getattr(model, "node_pred_keys", ())
        )
        if self.predicts_vertex:
            assert model is not None
            self.provides = self.provides | {
                "particle_vertex_proposals",
                "particle_interaction_ids",
            }
            if bool(getattr(model.node_encoder, "add_points", False)):
                self.provides = self.provides | {
                    "particle_vertex_start_points",
                    "particle_vertex_end_points",
                }

    def forward(self, state: ChainState) -> StageResult:
        """Build interaction candidates and publish native GrapPA outputs.

        Parameters
        ----------
        state : ChainState
            State containing particle products and optional truth.

        Returns
        -------
        StageResult
            Interaction clusters and non-delegated GrapPA diagnostics.
        """
        particles: IndexBatch = state.require("particle_clusts", self.name)
        shapes: TensorBatch = state.require("particle_shapes", self.name)
        primaries: IndexBatch = state.require("particle_primaries", self.name)
        outputs: dict[str, Any] = {}
        native: dict[str, Any] = {}

        # Learned grouping may also publish particle-level task predictions.
        if self.mode == "grappa":
            if self.model is None:
                raise RuntimeError("Interaction GrapPA was not initialized.")
            interactions, _, _, _, native = self.operations.run_grappa(
                self.model,
                state.require("point_data", self.name).data,
                particles,
                shapes,
                self.model.node_type,
                primaries=primaries,
                clust_label=state.get("clust_label"),
                coord_label=state.get("coord_label"),
                ppn_points=state.get("ppn_points"),
                point_use_primary=True,
            )
            # Do not republish clusters or node heads owned by another stage.
            for key, value in native.items():
                if key == "clusts":
                    continue
                task = key.removeprefix("node_").removesuffix("_pred")
                if key.startswith("node_") and self.task_modes.get(task) == "image":
                    continue
                outputs[f"particle_{key}"] = value
        else:
            # Truth mode groups reconstructed particles by interaction label.
            clust_label: ClusterLabelBatch | None = state.get("clust_label")
            if clust_label is None:
                raise ValueError(
                    "Label interaction aggregation requires `clust_label`."
                )
            interactions, _, _, _ = self.operations.group_labels(
                clust_label,
                particles,
                shapes,
            )

        products = {"interaction_clusts": interactions}
        if self.predicts_vertex:
            products.update(
                {
                    "particle_vertex_proposals": native["node_vertex_pred"],
                    "particle_interaction_ids": native["group_pred"],
                }
            )
            for key in ("start_points", "end_points"):
                if key in native:
                    products[f"particle_vertex_{key}"] = native[key]
        outputs.update(products)
        return StageResult(products, outputs)


class GrapPALossStage(ChainLossStage):
    """Route one namespaced GrapPA result to its native loss.

    Public prefixes identify which particle or interaction path produced an
    output. The adapter removes that namespace while preserving the standalone
    :class:`GrapPALoss` interface.
    """

    def __init__(self, name: str, prefix: str, loss: GrapPALoss) -> None:
        """Initialize a namespaced GrapPA loss adapter.

        Parameters
        ----------
        name : str
            Stage name.
        prefix : str
            Public output prefix removed before native loss evaluation.
        loss : GrapPALoss
            Native graph objective.
        """
        super().__init__(name)
        self.prefix = prefix
        self.loss = loss

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate GrapPA loss for one aggregation path.

        Parameters
        ----------
        data : dict
            Chain outputs and graph truth products.

        Returns
        -------
        dict
            Native GrapPA loss metrics.
        """
        clust_label = data.get("clust_label_adapt", data.get("clust_label"))
        if clust_label is None:
            raise ValueError("GrapPA loss requires `clust_label`.")
        # Restore standalone GrapPA output names from the public namespace.
        native = {
            key.removeprefix(self.prefix): value
            for key, value in data.items()
            if key.startswith(self.prefix)
        }
        return self.loss(
            clust_label=clust_label,
            coord_label=data.get("coord_label"),
            graph_label=data.get("graph_label"),
            **native,
        )


def _build_grappa(
    key: str,
    config: dict[str, Any] | None,
    owner: Any,
) -> GrapPA:
    """Construct and register one group-producing GrapPA.

    Parameters
    ----------
    key : str
        Module and configuration-block name.
    config : dict, optional
        Native GrapPA configuration.
    owner : torch.nn.Module
        Full-chain model that owns checkpoint parameters.

    Returns
    -------
    GrapPA
        Configured native graph model.
    """
    if not isinstance(config, dict):
        raise ValueError(f"Enabled GrapPA path requires a `{key}` block.")
    model = GrapPA(config)
    if not model.make_groups:
        raise ValueError(f"`{key}` must configure `make_groups: true`.")
    owner.add_module(key, model)
    return model


def build_particle_aggregation_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build separate or joint particle aggregation paths.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Aggregation modes and native GrapPA blocks.
    owner : torch.nn.Module
        Full-chain model that owns native graph modules.

    Returns
    -------
    ChainStage
        Configured particle aggregation adapter.
    """
    modes = {
        "shower": config.get("shower_aggregation"),
        "track": config.get("track_aggregation"),
        "particle": config.get("particle_aggregation"),
    }
    valid = {None, "skip", "grappa", "label"}
    for path, mode in modes.items():
        if mode not in valid:
            raise ValueError(f"Unknown {path} aggregation mode `{mode}`.")
    shower_primary = config.get("shower_primary")
    if shower_primary == "grappa" and modes["shower"] != "grappa":
        raise ValueError(
            "GrapPA shower-primary tagging requires GrapPA shower aggregation."
        )
    if modes["particle"] is not None and (
        modes["shower"] is not None or modes["track"] is not None
    ):
        raise ValueError(
            "Use joint particle aggregation or separate shower/track paths, not both."
        )

    # Instantiate only learned paths; label and skip modes own no modules.
    models = {}
    for path, mode in modes.items():
        if mode == "grappa":
            models[path] = _build_grappa(
                f"grappa_{path}",
                config.get(f"grappa_{path}"),
                owner,
            )
    operations = AggregationOperations(config.get("predict_points"))
    return ParticleAggregationStage(name, modes, models, operations)


def build_interaction_aggregation_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build interaction aggregation using GrapPA or truth labels.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Interaction mode, GrapPA block, and particle task ownership.
    owner : torch.nn.Module
        Full-chain model that owns the native graph module.

    Returns
    -------
    ChainStage
        Configured interaction aggregation adapter.
    """
    mode = config.get("mode")
    if not isinstance(mode, str):
        raise ValueError("Interaction aggregation requires a string `mode`.")

    model = None
    if mode == "grappa":
        model = _build_grappa("grappa_inter", config.get("grappa_inter"), owner)
    operations = AggregationOperations(config.get("predict_points"))
    task_modes = config.get("task_modes")
    if task_modes is not None and not isinstance(task_modes, dict):
        raise TypeError("Interaction `task_modes` must be a mapping.")
    return InteractionAggregationStage(name, mode, model, operations, task_modes)


def build_particle_aggregation_loss(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainLossStage | None:
    """Build adapters for configured particle-level GrapPA objectives.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Mapping of aggregation paths to resolved loss blocks.
    owner : torch.nn.Module
        Full-chain loss module that owns native objectives.

    Returns
    -------
    ChainLossStage or None
        Composite GrapPA adapter, or ``None`` without supervision.
    """
    loss_configs = config.get("loss") or {}
    if not isinstance(loss_configs, dict):
        raise TypeError("GrapPA loss configuration must be a mapping.")
    stages = []
    prefixes = {
        "shower": "shower_fragment_",
        "track": "track_fragment_",
        "particle": "fragment_",
    }
    for path, loss_config in loss_configs.items():
        if loss_config is None:
            continue
        if not isinstance(loss_config, dict):
            raise TypeError(f"GrapPA `{path}` loss must be a mapping.")
        loss = GrapPALoss(loss_config)
        owner.add_module(f"grappa_{path}_loss", loss)
        # The enclosing full-chain loss supplies the parent stage namespace.
        stages.append(GrapPALossStage(path, prefixes[path], loss))
    if not stages:
        return None
    return CompositeLossStage(name, stages)


def build_interaction_aggregation_loss(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainLossStage | None:
    """Build the interaction-level GrapPA objective when configured.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Interaction model context, task ownership, and resolved loss block.
    owner : torch.nn.Module
        Full-chain loss module that owns the native objective.

    Returns
    -------
    ChainLossStage or None
        Interaction GrapPA adapter, or ``None`` without supervision.
    """
    loss_config = config.get("loss")
    if loss_config is None:
        return None
    if not isinstance(loss_config, dict):
        raise TypeError("Interaction GrapPA loss must be a mapping.")

    # Image-owned particle tasks do not publish the corresponding GrapPA node
    # logits. Reject stale duplicate losses here instead of failing in forward.
    task_modes = config.get("task_modes") or {}
    image_tasks = {task for task, mode in task_modes.items() if mode == "image"}
    node_loss = loss_config.get("node_loss")
    if image_tasks and isinstance(node_loss, dict):
        if "name" in node_loss:
            raise ValueError(
                "A single interaction GrapPA node loss is ambiguous when particle "
                "tasks are delegated to the image provider. Configure named node "
                "losses and omit image-owned tasks."
            )
        conflicts = image_tasks.intersection(node_loss)
        if conflicts:
            tasks = ", ".join(sorted(conflicts))
            raise ValueError(
                "Interaction GrapPA loss still configures image-owned particle "
                f"task(s): {tasks}. Remove those node-loss blocks."
            )

    loss = GrapPALoss(loss_config)
    owner.add_module("grappa_inter_loss", loss)
    return GrapPALossStage(name, "particle_", loss)


class CompositeLossStage(ChainLossStage):
    """Combine several objectives owned by one logical chain stage.

    This is primarily used for independent shower and track GrapPA paths whose
    metrics must remain distinguishable but contribute one provider summary.
    """

    def __init__(self, name: str, stages: list[ChainLossStage]) -> None:
        """Initialize an ordered collection of loss adapters.

        Parameters
        ----------
        name : str
            Parent stage name.
        stages : list of ChainLossStage
            Child loss adapters.
        """
        super().__init__(name)
        self.stages = stages

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate child objectives and namespace their metrics.

        Parameters
        ----------
        data : dict
            Chain outputs and truth products.

        Returns
        -------
        dict
            Summed loss, mean accuracy, objective count, and child metrics.
        """
        result: dict[str, Any] = {"loss": 0.0, "accuracy": 0.0, "num_losses": 0}

        # Child adapters retain their own names so path diagnostics cannot
        # collide within the parent provider.
        for stage in self.stages:
            child = stage(data)
            result["loss"] = result["loss"] + child["loss"]
            result["accuracy"] += float(child.get("accuracy", 0.0))
            result["num_losses"] += 1
            result.update(
                {
                    f"{stage.name}_{key}": value
                    for key, value in child.items()
                    if key not in {"loss", "accuracy"}
                }
            )
        result["accuracy"] /= result["num_losses"]
        return result


PARTICLE_PROVIDER_SPEC = register_provider(
    ProviderSpec(
        "particle_aggregation",
        build_particle_aggregation_stage,
        build_particle_aggregation_loss,
    )
)
INTERACTION_PROVIDER_SPEC = register_provider(
    ProviderSpec(
        "interaction_aggregation",
        build_interaction_aggregation_stage,
        build_interaction_aggregation_loss,
    )
)
