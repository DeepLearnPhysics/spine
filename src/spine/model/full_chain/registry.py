"""Lazy registry of full-chain provider builders."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from .stage import ChainLossStage, ChainStage

StageBuilder = Callable[[str, dict[str, Any], Any], ChainStage]
LossBuilder = Callable[[str, dict[str, Any], Any], ChainLossStage | None]

__all__ = ["ProviderSpec", "provider_spec", "register_provider"]


class ProviderSpec:
    """Pair network and optional loss builders for one chain provider.

    Parameters
    ----------
    name : str
        Unique provider name referenced by full-chain configurations.
    stage : callable
        Builder for the network-side stage adapter.
    loss : callable, optional
        Builder for the corresponding loss adapter.
    """

    def __init__(
        self,
        name: str,
        stage: StageBuilder,
        loss: LossBuilder | None = None,
    ) -> None:
        """Initialize a provider specification.

        Parameters
        ----------
        name : str
            Unique provider name.
        stage : callable
            Network-stage builder.
        loss : callable, optional
            Loss-stage builder.
        """
        self.name = name
        self.stage = stage
        self.loss = loss


_PROVIDERS: dict[str, ProviderSpec] = {}
_BUILTIN_MODULES = {
    "deghost": "spine.model.full_chain.providers.deghost",
    "segmentation": "spine.model.full_chain.providers.segmentation",
    "fragmentation": "spine.model.full_chain.providers.fragmentation",
    "particle_aggregation": "spine.model.full_chain.providers.aggregation",
    "interaction_aggregation": "spine.model.full_chain.providers.aggregation",
    "interaction_vertexing": "spine.model.full_chain.providers.vertexing",
    "particle_image": "spine.model.full_chain.providers.image",
    "calibration": "spine.model.full_chain.providers.calibration",
    "track_breaking": "spine.model.full_chain.providers.transform.track_breaking",
}


def register_provider(spec: ProviderSpec) -> ProviderSpec:
    """Register one provider, rejecting accidental name collisions.

    Parameters
    ----------
    spec : ProviderSpec
        Provider definition to register.

    Returns
    -------
    ProviderSpec
        The registered definition, allowing use as a decorator helper.
    """
    if spec.name in _PROVIDERS and _PROVIDERS[spec.name] is not spec:
        raise ValueError(f"Full-chain provider `{spec.name}` is already registered.")
    _PROVIDERS[spec.name] = spec
    return spec


def provider_spec(name: str) -> ProviderSpec:
    """Resolve a built-in, registered or import-path provider specification.

    Parameters
    ----------
    name : str
        Built-in provider name or ``module:attribute`` import path.

    Returns
    -------
    ProviderSpec
        Resolved provider definition.

    Raises
    ------
    TypeError
        If an import path resolves to an incompatible object.
    ValueError
        If no registered or importable provider matches ``name``.
    """
    # Built-ins are imported on demand to avoid eagerly importing every model
    # implementation when the full-chain package is discovered.
    if name not in _PROVIDERS and name in _BUILTIN_MODULES:
        import_module(_BUILTIN_MODULES[name])

    # Import paths provide a zero-touch extension point for external packages.
    if name not in _PROVIDERS and ":" in name:
        module_name, attribute = name.split(":", 1)
        candidate = getattr(import_module(module_name), attribute)
        if not isinstance(candidate, ProviderSpec):
            raise TypeError(f"Imported provider `{name}` is not a ProviderSpec.")
        register_provider(candidate)
        return candidate

    try:
        return _PROVIDERS[name]
    except KeyError as err:
        valid = ", ".join(sorted(set(_BUILTIN_MODULES).union(_PROVIDERS)))
        raise ValueError(
            f"Unknown full-chain provider `{name}`. Available providers: {valid}."
        ) from err
