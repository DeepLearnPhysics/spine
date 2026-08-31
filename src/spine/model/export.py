"""Compose configured component checkpoints into one model artifact."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from spine.config import normalize_config, to_inference_config

from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManifest,
    checkpoint_sha256,
    save_checkpoint,
)
from .manager import ModelManager

__all__ = ["export_model_weights"]


def _remove_weight_paths(value: Any) -> Any:
    """Return a configuration copy without external checkpoint dependencies."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key not in {"weight_path", "weight_list"}:
                result[key] = _remove_weight_paths(item)
        return result
    if isinstance(value, list):
        return [_remove_weight_paths(item) for item in value]
    return deepcopy(value)


def export_model_weights(
    cfg: Mapping[str, Any],
    path: str | os.PathLike[str],
) -> str:
    """Build and serialize a strictly populated inference model on CPU.

    Component ``weight_path`` entries are loaded through
    :class:`ModelManager`, including its module namespace remapping. Every
    persistent entry in the resulting model state must have been supplied by
    at least one checkpoint; constructor initialization is never silently
    exported.

    Parameters
    ----------
    cfg : mapping
        Complete resolved SPINE configuration containing a model block and
        global or module-specific checkpoint paths.
    path : path-like
        Destination for the composed checkpoint.

    Returns
    -------
    str
        SHA-256 digest of the exported checkpoint.

    Raises
    ------
    KeyError
        If the configuration has no model block.
    ValueError
        If the model has no checkpoint inputs or any state entry remains
        populated only by its constructor.
    """
    resolved = normalize_config(deepcopy(dict(cfg)))
    if "model" not in resolved:
        raise KeyError("--export-weights requires a `model` block.")

    model_cfg = deepcopy(resolved["model"])
    if not isinstance(model_cfg, MutableMapping):
        raise TypeError("The `model` block must be a mapping.")
    if model_cfg.get("weight_list") is not None:
        raise ValueError("--export-weights does not support `model.weight_list`.")

    # Composition constructs only the authoritative network. Losses, training
    # state, data conversion and distributed wrappers are intentionally absent.
    model_cfg.pop("loss_input", None)
    model_cfg["to_numpy"] = False
    model_cfg.setdefault("dtype", resolved.get("base", {}).get("dtype", "float32"))
    manager = ModelManager(**model_cfg)
    if not manager.loaded_weight_sources:
        raise ValueError("--export-weights requires at least one checkpoint input.")

    model = getattr(manager.net, "module", manager.net)
    state_dict = model.state_dict()
    missing = sorted(set(state_dict).difference(manager.loaded_weight_keys))
    if missing:
        detail = ", ".join(missing[:10])
        if len(missing) > 10:
            detail += f", ... ({len(missing) - 10} more)"
        raise ValueError(
            "Cannot export weights because some model state was not supplied "
            f"by a checkpoint: {detail}"
        )

    destination = Path(path).expanduser().resolve()
    sources = []
    for record in manager.loaded_weight_sources:
        source_path = Path(record["path"]).expanduser().resolve()
        if source_path == destination:
            raise ValueError(
                "The export destination cannot overwrite an input checkpoint."
            )
        sources.append(
            {
                "module": record["module"],
                "model_name": record["model_name"],
                "path": str(source_path),
                "sha256": checkpoint_sha256(source_path),
                "keys": record["keys"],
            }
        )

    # Store a directly runnable inference configuration without retaining the
    # component files as runtime dependencies of the composed artifact.
    inference_cfg = to_inference_config(resolved, cpu=True)
    inference_cfg["model"] = _remove_weight_paths(inference_cfg["model"])
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": inference_cfg,
        "state_dict": state_dict,
        "weight_sources": sources,
    }
    checkpoint["manifest"] = CheckpointManifest.create(
        contents=tuple(sorted(checkpoint))
    ).to_dict()
    return save_checkpoint(checkpoint, destination)
