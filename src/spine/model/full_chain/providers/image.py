"""Object-image particle-task provider for the full reconstruction chain."""

from __future__ import annotations

from typing import Any

from spine.data import IndexBatch, TensorBatch
from spine.model.image import ImageLoss, ImageModel

from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult


class ParticleImageStage(ChainStage):
    """Run classification or regression heads on reconstructed particles.

    Reconstructed particle indexes are supplied explicitly to the standalone
    image model. Predictions use GrapPA-compatible public names so downstream
    constructors are independent of task implementation.
    """

    requires = frozenset({"data", "particle_clusts"})

    def __init__(
        self,
        name: str,
        model: ImageModel,
        heads: dict[str, str],
    ) -> None:
        """Initialize the object-image provider.

        Parameters
        ----------
        name : str
            Stage name.
        model : ImageModel
            Shared particle-image encoder and prediction heads.
        heads : dict
            Mapping from native image head names to canonical particle task
            names.
        """
        super().__init__(name)
        self.model = model
        self.heads = heads
        self.provides = frozenset(
            f"particle_{output_name}_pred" for output_name in heads.values()
        )

    def forward(self, state: ChainState) -> StageResult:
        """Encode reconstructed particles and publish task predictions.

        Parameters
        ----------
        state : ChainState
            State containing canonical voxel data and particle clusters.

        Returns
        -------
        StageResult
            Canonical particle predictions and GrapPA-compatible outputs.
        """
        data: TensorBatch = state.require("data", self.name)
        particles: IndexBatch = state.require("particle_clusts", self.name)
        native = self.model(data, objects=particles)

        # Reuse GrapPA's established particle output keys so constructors and
        # analysis code are indifferent to which task provider was selected.
        outputs: dict[str, Any] = {}
        products: dict[str, Any] = {}
        for head, output_name in self.heads.items():
            key = f"particle_node_{output_name}_pred"
            value = native[f"{head}_pred"]
            outputs[key] = value
            products[f"particle_{output_name}_pred"] = value
        if "features" in native:
            outputs["particle_image_features"] = native["features"]
        return StageResult(products, outputs)


class ParticleImageLossStage(ChainLossStage):
    """Route canonical particle-image predictions to :class:`ImageLoss`.

    Canonical task names make predictions interchangeable with GrapPA. This
    adapter reverses those aliases before evaluating the native image heads.
    """

    def __init__(self, name: str, loss: ImageLoss, heads: dict[str, str]) -> None:
        """Initialize the image-task loss adapter.

        Parameters
        ----------
        name : str
            Stage name.
        loss : ImageLoss
            Native multi-head image objective.
        heads : dict
            Native-to-canonical task-name mapping.
        """
        super().__init__(name)
        self.loss = loss
        self.heads = heads

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate all configured object-image objectives.

        Parameters
        ----------
        data : dict
            Particle clusters, labels, and canonical task predictions.

        Returns
        -------
        dict
            Combined image-task loss and per-head diagnostics.
        """
        particles = data.get("particle_clusts")
        if particles is None:
            raise ValueError("Particle-image loss requires `particle_clusts`.")
        # Restore native head names expected by the standalone ImageLoss.
        native = dict(data)
        for head, output_name in self.heads.items():
            key = f"particle_node_{output_name}_pred"
            if key not in data:
                raise ValueError(f"Particle-image loss requires `{key}`.")
            native[f"{head}_pred"] = data[key]
        return self.loss(particles, **native)


def _image_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate the nested image-model configuration.

    Parameters
    ----------
    config : dict
        Provider configuration containing ``image`` or ``image_particle``.

    Returns
    -------
    dict
        Independent image-model configuration using explicit objects.
    """
    image = config.get("image", config.get("image_particle"))
    if not isinstance(image, dict):
        raise ValueError("Particle-image tasks require an `image` block.")
    result = dict(image)
    object_config = result.get("objects")
    if object_config is None:
        objects = {}
    elif isinstance(object_config, dict):
        objects = dict(object_config)
    else:
        raise TypeError("Particle-image `objects` configuration must be a mapping.")
    source = objects.setdefault("source", "explicit")
    if source != "explicit":
        raise ValueError("Full-chain particle images require `source: explicit`.")
    result["objects"] = objects
    return result


def _head_names(image: dict[str, Any]) -> dict[str, str]:
    """Map image heads onto established reconstruction task names.

    Parameters
    ----------
    image : dict
        Validated image-model configuration.

    Returns
    -------
    dict
        Native head name to canonical task-name mapping.
    """
    heads = image.get("heads")
    if not isinstance(heads, dict):
        raise ValueError("Image `heads` must be a mapping.")

    names: dict[str, str] = {}
    aliases: dict[str, str] = {"pid": "type", "orientation": "orient"}
    for head in heads:
        if not isinstance(head, str) or not head:
            raise ValueError("Image head names must be nonempty strings.")
        names[head] = aliases.get(head, head)
    return names


def build_particle_image_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build a shared image encoder with particle prediction heads.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Resolved image-model block and optional legacy task modes.
    owner : torch.nn.Module
        Full-chain model that owns the native image module.

    Returns
    -------
    ChainStage
        Particle-image task adapter.
    """
    image = _image_config(config)
    model = ImageModel(image)
    owner.add_module("image_particle", model)
    heads = _head_names(image)

    # In legacy mode, verify that every task delegated away from GrapPA has a
    # corresponding image head. Native plans may use arbitrary named heads.
    task_heads = {
        "particle_identification": "type",
        "primary_identification": "primary",
        "orientation_identification": "orient",
    }
    requested = {
        output_name
        for task, output_name in task_heads.items()
        if config.get(task) == "image"
    }
    missing = requested.difference(heads.values())
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Particle-image configuration is missing heads: {names}.")
    return ParticleImageStage(name, model, heads)


def build_particle_image_loss(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainLossStage | None:
    """Build particle-level classification and regression objectives.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Resolved image-model and image-loss blocks.
    owner : torch.nn.Module
        Full-chain loss module that owns the native objective.

    Returns
    -------
    ChainLossStage or None
        Image-task loss adapter, or ``None`` without supervision.
    """
    loss_config = config.get("loss")
    if loss_config is None:
        return None
    if not isinstance(loss_config, dict):
        raise TypeError("Particle-image loss configuration must be a mapping.")
    image = _image_config(config)
    loss = ImageLoss(image, loss_config)
    owner.add_module("image_particle_loss", loss)
    heads = _head_names(image)
    return ParticleImageLossStage(name, loss, heads)


PROVIDER_SPEC = register_provider(
    ProviderSpec(
        "particle_image", build_particle_image_stage, build_particle_image_loss
    )
)
