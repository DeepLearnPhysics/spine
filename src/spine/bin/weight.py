"""Apply command-line overrides for model-module weights."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

__all__ = ["apply_module_weight_overrides", "parse_module_weights"]


def parse_module_weights(values: list[str] | None) -> dict[str, str]:
    """Parse ``MODULE=PATH`` weight assignments.

    Parameters
    ----------
    values : list[str], optional
        Module-qualified checkpoint paths supplied by ``--module-weight``.

    Returns
    -------
    dict[str, str]
        Module names mapped to checkpoint paths.

    Raises
    ------
    ValueError
        If an assignment is malformed or names a module more than once.
    """
    overrides: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(
                f"Invalid --module-weight value '{value}'. Expected 'MODULE=PATH'."
            )

        # Split once so checkpoint paths may contain additional equals signs.
        module, path = value.split("=", 1)
        if not module or not path:
            raise ValueError(
                f"Invalid --module-weight value '{value}'. Expected 'MODULE=PATH'."
            )
        if module in overrides:
            raise ValueError(
                f"Module '{module}' has multiple --module-weight assignments."
            )
        overrides[module] = path

    return overrides


def apply_module_weight_overrides(
    model_cfg: MutableMapping[str, Any],
    values: list[str] | None,
) -> None:
    """Apply module-specific checkpoint paths to a model configuration.

    Each assignment updates ``model.modules.<module>.weight_path``. Validation
    is completed for every assignment before the configuration is mutated, so
    an invalid invocation cannot leave a partially updated configuration.

    Parameters
    ----------
    model_cfg : MutableMapping
        Top-level ``model`` configuration to update in place.
    values : list[str], optional
        Module-qualified checkpoint paths supplied by ``--module-weight``.

    Raises
    ------
    KeyError
        If the model has no ``modules`` block or a named module does not exist.
    TypeError
        If the modules block or a selected module is not an inline mutable
        mapping.
    ValueError
        If an assignment is malformed or duplicated.
    """
    overrides = parse_module_weights(values)
    if not overrides:
        return

    if "modules" not in model_cfg:
        raise KeyError("--module-weight requires a `model.modules` block.")
    modules = model_cfg["modules"]
    if not isinstance(modules, MutableMapping):
        raise TypeError("The `model.modules` block must be an inline mapping.")

    # Resolve every target before applying any checkpoint paths.
    targets: dict[str, MutableMapping[str, Any]] = {}
    for module in overrides:
        if module not in modules:
            raise KeyError(
                f"Unknown --module-weight module '{module}'; no matching "
                "`model.modules` block was found."
            )
        module_cfg = modules[module]
        if not isinstance(module_cfg, MutableMapping):
            raise TypeError(
                f"The `model.modules.{module}` block must be an inline mapping."
            )
        targets[module] = module_cfg

    for module, path in overrides.items():
        targets[module]["weight_path"] = path
