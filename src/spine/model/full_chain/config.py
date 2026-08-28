"""Configuration parsing for ordered full-chain execution plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["StageConfig", "build_chain_plan", "get_chain_inputs"]


@dataclass(frozen=True)
class StageConfig:
    """Normalized configuration for one provider instance.

    Parameters
    ----------
    name : str
        Unique name of the stage within the execution plan.
    provider : str
        Registered provider name or import path used to build the stage.
    config : dict
        Provider-specific network configuration.
    loss_config : dict, optional
        Provider-specific objective configuration.
    """

    name: str
    provider: str
    config: dict[str, Any]
    loss_config: dict[str, Any] | None = None


def get_chain_inputs(chain: dict[str, Any]) -> frozenset[str]:
    """Return canonical products declared as external chain inputs.

    Parameters
    ----------
    chain : dict
        Native ordered-stage configuration.

    Returns
    -------
    frozenset of str
        Product names supplied by the caller before provider execution.

    Raises
    ------
    TypeError
        If the chain or input collection has an invalid type.
    ValueError
        If inputs are used with the legacy schema or contain invalid names.
    """
    if not isinstance(chain, dict):
        raise TypeError("Full-chain `chain` configuration must be a mapping.")

    if "inputs" not in chain:
        return frozenset()
    if "stages" not in chain:
        raise ValueError("Full-chain `inputs` require the native `stages` schema.")

    inputs = chain["inputs"]
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Full-chain `inputs` must be a list of product names.")
    if not all(isinstance(name, str) and name for name in inputs):
        raise ValueError("Full-chain input names must be nonempty strings.")
    if len(set(inputs)) != len(inputs):
        raise ValueError("Full-chain input names must be unique.")

    return frozenset(inputs)


def _new_chain_plan(
    stages: Any,
    modules: dict[str, Any],
    require_losses: bool,
) -> list[StageConfig]:
    """Normalize the native ordered-stage configuration.

    Parameters
    ----------
    stages : object
        Raw value associated with ``chain.stages``.
    modules : dict
        Named model and loss configuration blocks.
    require_losses : bool
        Whether references to absent loss blocks are errors.

    Returns
    -------
    list of StageConfig
        Validated stages in execution order.

    Raises
    ------
    TypeError
        If a stage descriptor or referenced block has an invalid type.
    ValueError
        If the plan is empty, names are duplicated, or a referenced block is
        unavailable.
    """
    if not isinstance(stages, list) or not stages:
        raise ValueError("Full-chain `stages` must be a nonempty list.")

    result = []
    names = set()
    for stage in stages:
        # Separate the provider identity from its inline configuration.
        if not isinstance(stage, dict):
            raise TypeError("Each full-chain stage must be a mapping.")
        descriptor = dict(stage)
        try:
            name = descriptor.pop("name")
            provider = descriptor.pop("provider")
        except KeyError as err:
            raise ValueError(
                "Each full-chain stage requires `name` and `provider`."
            ) from err
        if not isinstance(name, str) or not name:
            raise ValueError("Full-chain stage names must be nonempty strings.")
        if not isinstance(provider, str) or not provider:
            raise ValueError("Full-chain provider names must be nonempty strings.")
        if name in names:
            raise ValueError(f"Duplicate full-chain stage name `{name}`.")
        names.add(name)

        # Normalize references to sibling model and loss blocks.
        inline_config = descriptor.pop("config", {})
        if not isinstance(inline_config, dict):
            raise TypeError(f"Stage `{name}` `config` must be a mapping.")
        uses = descriptor.pop("uses", ())
        if isinstance(uses, str):
            uses = (uses,)
        if not isinstance(uses, (list, tuple)) or not all(
            isinstance(key, str) for key in uses
        ):
            raise TypeError(f"Stage `{name}` `uses` must contain block names.")
        loss_reference = descriptor.pop("loss", None)
        if loss_reference is not None and not isinstance(loss_reference, (str, dict)):
            raise TypeError(
                f"Stage `{name}` loss must be a block name, mapping or null."
            )

        # Referenced module blocks retain their top-level names so provider
        # builders receive the same shape as standalone model constructors.
        config = {}
        for key in uses:
            if key not in modules:
                raise ValueError(f"Stage `{name}` references missing block `{key}`.")
            config[key] = modules[key]
        config.update(inline_config)
        config.update(descriptor)

        # Loss may name one block or map several provider-owned objectives to
        # independent blocks, as with shower and track GrapPA paths.
        loss_config = None
        if isinstance(loss_reference, str):
            if require_losses and loss_reference not in modules:
                raise ValueError(
                    f"Stage `{name}` references missing loss `{loss_reference}`."
                )
            if loss_reference in modules:
                candidate = modules[loss_reference]
                if not isinstance(candidate, dict):
                    raise TypeError(f"Loss block `{loss_reference}` must be a mapping.")
                loss_config = dict(candidate)
        elif isinstance(loss_reference, dict):
            loss_config = {}
            for key, block_name in loss_reference.items():
                if not isinstance(key, str) or not isinstance(block_name, str):
                    raise TypeError(
                        f"Stage `{name}` loss mapping must map names to block names."
                    )
                if require_losses and block_name not in modules:
                    raise ValueError(
                        f"Stage `{name}` references missing loss `{block_name}`."
                    )
                if block_name in modules:
                    candidate = modules[block_name]
                    if not isinstance(candidate, dict):
                        raise TypeError(f"Loss block `{block_name}` must be a mapping.")
                    loss_config[key] = candidate
            if not loss_config:
                loss_config = None

        result.append(StageConfig(name, provider, config, loss_config))

    return result


def _legacy_chain_plan(
    chain: dict[str, Any],
    modules: dict[str, Any],
) -> list[StageConfig]:
    """Translate the historical mode matrix into the native stage plan.

    Parameters
    ----------
    chain : dict
        Historical task-to-mode mapping.
    modules : dict
        Named model and loss configuration blocks.

    Returns
    -------
    list of StageConfig
        Native provider stages in historical execution order.

    Raises
    ------
    ValueError
        If no stage is enabled or calibration references an unavailable
        target stage.
    """
    config = dict(chain)
    stages: list[StageConfig] = []

    # Translate voxel preprocessing and semantic reconstruction stages.
    deghosting = config.get("deghosting")
    charge_rescaling = config.get("charge_rescaling")
    if deghosting is not None or charge_rescaling is not None:
        stages.append(
            StageConfig(
                "deghosting",
                "deghost",
                {
                    "mode": deghosting,
                    "charge_rescaling": charge_rescaling,
                    "uresnet_deghost": modules.get("uresnet_deghost"),
                },
                modules.get("uresnet_deghost_loss"),
            )
        )

    segmentation = config.get("segmentation")
    point_proposal = config.get("point_proposal")
    if segmentation is not None or point_proposal is not None:
        stages.append(
            StageConfig(
                "segmentation",
                "segmentation",
                {
                    "mode": segmentation,
                    "point_proposal": point_proposal,
                    "uresnet": modules.get("uresnet"),
                    "uresnet_ppn": modules.get("uresnet_ppn"),
                    "adapt_labels": modules.get("adapt_labels"),
                    "predict_points": modules.get("predict_points"),
                },
                modules.get("uresnet_ppn_loss") or modules.get("uresnet_loss"),
            )
        )

    fragmentation = config.get("fragmentation")
    if fragmentation is not None:
        fragmentation_losses = {
            key: value
            for key, value in {
                "spice": modules.get("spice_loss"),
                "graph_spice": modules.get("graph_spice_loss"),
            }.items()
            if value is not None
        }
        stages.append(
            StageConfig(
                "fragmentation",
                "fragmentation",
                {
                    "mode": fragmentation,
                    "dbscan": modules.get("dbscan"),
                    "spice": modules.get("spice"),
                    "graph_spice": modules.get("graph_spice"),
                },
                fragmentation_losses or None,
            )
        )

    # Particle construction can use separate shower/track paths or one joint
    # path; the provider performs the detailed exclusivity validation.
    aggregation_modes = {
        "shower_aggregation": config.get("shower_aggregation"),
        "shower_primary": config.get("shower_primary"),
        "track_aggregation": config.get("track_aggregation"),
        "particle_aggregation": config.get("particle_aggregation"),
    }
    if any(mode is not None for mode in aggregation_modes.values()):
        stages.append(
            StageConfig(
                "particle_aggregation",
                "particle_aggregation",
                aggregation_modes
                | {
                    "grappa_shower": modules.get("grappa_shower"),
                    "grappa_track": modules.get("grappa_track"),
                    "grappa_particle": modules.get("grappa_particle"),
                    "predict_points": modules.get("predict_points"),
                },
                {
                    "shower": modules.get("grappa_shower_loss"),
                    "track": modules.get("grappa_track_loss"),
                    "particle": modules.get("grappa_particle_loss"),
                },
            )
        )

    # Insert an object-image task provider only when legacy modes delegate at
    # least one particle prediction away from interaction GrapPA.
    image_tasks = {
        "particle_identification": config.get("particle_identification"),
        "primary_identification": config.get("primary_identification"),
        "orientation_identification": config.get("orientation_identification"),
    }
    if any(mode == "image" for mode in image_tasks.values()):
        stages.append(
            StageConfig(
                "particle_image",
                "particle_image",
                image_tasks | {"image": modules.get("image_particle")},
                modules.get("image_particle_loss"),
            )
        )

    # Interaction aggregation remains independent of particle task ownership.
    inter_aggregation = config.get("inter_aggregation")
    if inter_aggregation is not None:
        stages.append(
            StageConfig(
                "interaction_aggregation",
                "interaction_aggregation",
                {
                    "mode": inter_aggregation,
                    "grappa_inter": modules.get("grappa_inter"),
                    "predict_points": modules.get("predict_points"),
                    "task_modes": {
                        "type": config.get("particle_identification"),
                        "primary": config.get("primary_identification"),
                        "orient": config.get("orientation_identification"),
                    },
                },
                modules.get("grappa_inter_loss"),
            )
        )

    # Vertex reduction follows interaction aggregation because both supported
    # modes publish exactly one prediction for each reconstructed interaction.
    vertexing = config.get("vertexing")
    if vertexing is not None:
        vertexing_config = modules.get("interaction_vertexing")
        if vertexing_config is None:
            vertexing_config = {}
        if not isinstance(vertexing_config, dict):
            raise TypeError("`interaction_vertexing` must be a mapping.")
        stages.append(
            StageConfig(
                "interaction_vertexing",
                "interaction_vertexing",
                {"mode": vertexing, **vertexing_config},
            )
        )

    # Historical calibration named a target stage rather than occupying an
    # explicit position. Resolve aliases and insert it immediately beforehand.
    calibration = config.get("calibration")
    if calibration is not None:
        calibration_config = modules.get("calibration")
        if not isinstance(calibration_config, dict):
            raise ValueError(
                "Enabled calibration requires a `calibration` configuration block."
            )
        calibration_config = dict(calibration_config)
        try:
            target = calibration_config.pop("stage")
        except KeyError as err:
            raise ValueError("Calibration configuration requires `stage`.") from err
        aliases = {
            "particle_classification": "particle_image",
            "particle_identification": "particle_image",
            "inter_aggregation": "interaction_aggregation",
        }
        target = aliases.get(target, target)
        positions = [
            index for index, stage in enumerate(stages) if stage.name == target
        ]
        if not positions:
            raise ValueError(f"Calibration target stage `{target}` is not enabled.")
        stages.insert(
            positions[0],
            StageConfig(
                f"calibration_before_{target}",
                "calibration",
                {"mode": calibration, "calibration": calibration_config},
            ),
        )

    if not stages:
        raise ValueError("The full chain must enable at least one stage.")
    return stages


def build_chain_plan(
    chain: dict[str, Any],
    modules: dict[str, Any],
    require_losses: bool = False,
) -> list[StageConfig]:
    """Build a normalized ordered plan from native or legacy configuration.

    Parameters
    ----------
    chain : dict
        Native ordered-stage configuration or historical mode matrix.
    modules : dict
        Named model and loss blocks available to stage ``uses`` and ``loss``
        references.
    require_losses : bool, default False
        If ``True``, reject references to missing loss blocks. Network
        construction leaves this disabled because the model manager removes
        loss blocks before initializing the network.

    Returns
    -------
    list of StageConfig
        Normalized provider execution plan.

    Raises
    ------
    TypeError
        If ``chain`` is not a mapping.
    ValueError
        If native configuration mixes ordered stages with legacy options.
    """
    if not isinstance(chain, dict):
        raise TypeError("Full-chain `chain` configuration must be a mapping.")
    get_chain_inputs(chain)

    # Presence of `stages` unambiguously selects the native ordered schema.
    if "stages" in chain:
        unknown = set(chain).difference({"inputs", "stages"})
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Native chain configuration has unknown keys: {names}.")
        return _new_chain_plan(chain["stages"], modules, require_losses)
    return _legacy_chain_plan(chain, modules)
