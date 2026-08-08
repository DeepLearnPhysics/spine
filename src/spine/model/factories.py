"""Factories for top-level machine-learning models."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .registry import ModelSpec

__all__ = ["model_dict", "model_factory", "model_names", "model_spec"]

_MODEL_MODULES: dict[str, str] = {
    "full_chain": "spine.model.full_chain",
    "graph_spice": "spine.model.graph_spice",
    "grappa": "spine.model.grappa",
    "image": "spine.model.image",
    "spice": "spine.model.spice",
    "uresnet": "spine.model.uresnet",
    "uresnet_bayes": "spine.model.uresnet.bayes",
    "uresnet_ppn": "spine.model.uresnet.ppn",
}


def model_names() -> tuple[str, ...]:
    """Return supported model names without importing model implementations."""

    return tuple(_MODEL_MODULES)


def model_spec(name: str) -> ModelSpec:
    """Load the specification for one supported model lazily."""

    if name not in _MODEL_MODULES:
        valid_names = ", ".join(model_names())
        raise ValueError(
            f"Unknown model name `{name}`. Available models: {valid_names}"
        )

    module = import_module(_MODEL_MODULES[name])
    try:
        spec = module.MODEL_SPEC
    except AttributeError as err:
        raise RuntimeError(
            f"Model module `{module.__name__}` does not define `MODEL_SPEC`."
        ) from err

    if not isinstance(spec, ModelSpec):
        raise TypeError(f"`{module.__name__}.MODEL_SPEC` must be a ModelSpec.")
    if spec.name != name:
        raise ValueError(
            f"Model specification name `{spec.name}` does not match registry "
            f"name `{name}`."
        )

    return spec


def model_factory(name: str) -> tuple[type[Any], type[Any] | None]:
    """Return the network and loss classes associated with a model name."""

    spec = model_spec(name)
    return spec.network, spec.loss


def model_dict() -> dict[str, tuple[type[Any], type[Any] | None]]:
    """Return all supported model/loss pairs.

    This compatibility helper imports every supported model. Prefer
    :func:`model_names` when only discovery is required.
    """

    return {name: model_factory(name) for name in model_names()}
