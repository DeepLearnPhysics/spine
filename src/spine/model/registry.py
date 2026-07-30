"""Definitions shared by the top-level model registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Pair a model network with its associated loss implementation."""

    name: str
    network: type[Any]
    loss: type[Any] | None = None
