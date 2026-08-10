"""Runtime configuration normalization helpers."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

__all__ = ["normalize_config"]


def normalize_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy runtime configuration without mutating the input.

    Parameters
    ----------
    cfg : mapping
        Complete user configuration.

    Returns
    -------
    dict
        Independent normalized configuration mapping.

    Raises
    ------
    ValueError
        If training is configured in both supported locations.

    Warns
    -----
    FutureWarning
        If the deprecated ``base.train`` location is normalized.
    """
    # Copy before relocating legacy blocks
    cfg = deepcopy(dict(cfg))
    base = cfg.get("base")
    if not isinstance(base, Mapping) or "train" not in base:
        return cfg
    if "train" in cfg:
        raise ValueError(
            "Configure `train` either at top level or in `base`, not both."
        )

    # Promote legacy training configuration to its canonical location
    base = dict(base)
    cfg["train"] = base.pop("train")
    cfg["base"] = base
    warnings.warn(
        "`base.train` is deprecated; move `train` to the top level.",
        FutureWarning,
        stacklevel=2,
    )
    return cfg
