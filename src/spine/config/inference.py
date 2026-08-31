"""Configuration transforms for inference workflows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from warnings import warn

import yaml

from .normalize import normalize_config

__all__ = ["get_inference_cfg", "to_inference_config"]


def to_inference_config(
    cfg: Mapping[str, Any],
    *,
    file_keys: str | list[str] | None = None,
    weight_path: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cpu: bool = False,
) -> dict[str, Any]:
    """Convert a training configuration into an inference configuration.

    The transform removes training-only scheduling, switches loader-backed
    input to deterministic sequential traversal, and preserves all model and
    downstream reconstruction configuration. Explicit keyword overrides are
    applied after the structural conversion.

    Parameters
    ----------
    cfg : mapping
        Complete SPINE configuration.
    file_keys : str or list of str, optional
        Replacement input file selection for a loader-backed configuration.
    weight_path : str, optional
        Model checkpoint or checkpoint pattern to load.
    batch_size : int, optional
        Global inference batch size.
    num_workers : int, optional
        Number of loader worker processes.
    cpu : bool, default False
        If ``True``, request CPU execution.

    Returns
    -------
    dict
        Independent inference configuration. The input mapping is not mutated.

    Raises
    ------
    ValueError
        If a loader-specific override is requested without a loader.
    """
    result = normalize_config(deepcopy(dict(cfg)))
    base = result.setdefault("base", {})

    # Training and checkpoint-bound validation do not participate in ordinary
    # inference. Replace an epoch schedule with one complete input traversal.
    result.pop("train", None)
    result.pop("validation", None)
    base.pop("epochs", None)
    base.setdefault("iterations", -1)

    loader = result.get("io", {}).get("loader")
    loader_overrides = any(
        value is not None for value in (file_keys, batch_size, num_workers)
    )
    if loader is None and loader_overrides:
        raise ValueError("Loader overrides require an `io.loader` configuration.")

    if loader is not None:
        loader = dict(loader)
        loader["shuffle"] = False
        loader.pop("sampler", None)
        if batch_size is not None:
            loader.pop("minibatch_size", None)
            loader["batch_size"] = batch_size
        if num_workers is not None:
            loader["num_workers"] = num_workers
        if file_keys is not None:
            dataset = dict(loader.get("dataset", {}))
            dataset["file_keys"] = file_keys
            dataset.pop("file_list", None)
            loader["dataset"] = dataset
        result["io"]["loader"] = loader

    if weight_path is not None:
        if "model" not in result:
            raise ValueError("A weight override requires a `model` configuration.")
        result["model"]["weight_path"] = weight_path

    if cpu:
        base["world_size"] = 0

    # Event-level consumers require loader batches to be unwrapped. Raw model
    # inference remains batched unless the configuration asks for such output.
    io_writer = result.get("io", {}).get("writer")
    if io_writer is not None or any(key in result for key in ("build", "post", "ana")):
        base["unwrap"] = True

    return result


def get_inference_cfg(
    cfg: Mapping[str, Any] | str | Path,
    file_keys: str | list[str] | None = None,
    weight_path: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cpu: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper for :func:`to_inference_config`.

    New code should load configurations through :mod:`spine.config` and call
    :func:`to_inference_config` directly.

    Parameters
    ----------
    cfg : mapping, str or pathlib.Path
        Configuration mapping or path to a YAML configuration.
    file_keys : str or list[str], optional
        Replacement input files.
    weight_path : str, optional
        Model checkpoint or checkpoint pattern.
    batch_size : int, optional
        Global inference batch size.
    num_workers : int, optional
        Number of loader worker processes.
    cpu : bool, default False
        Request CPU execution.
    """
    warn(
        "`get_inference_cfg` is deprecated; use `to_inference_config`.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(cfg, (str, Path)):
        with open(cfg, "r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
    else:
        loaded = cfg

    return to_inference_config(
        loaded,
        file_keys=file_keys,
        weight_path=weight_path,
        batch_size=batch_size,
        num_workers=num_workers,
        cpu=cpu,
    )
