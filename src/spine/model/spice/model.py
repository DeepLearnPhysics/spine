"""SPICE spatial-embedding clustering model and objective."""

from __future__ import annotations

from typing import Any

from ..registry import ModelSpec
from .embedder import SPICEEmbedder
from .loss import SPICELoss as _SPICELoss

__all__ = ["SPICE", "SPICELoss"]


class SPICE(SPICEEmbedder):
    """Spatial embeddings for proposal-free instance clustering."""

    def __init__(self, spice: dict[str, Any]) -> None:
        """Initialize the SPICE model.

        Parameters
        ----------
        spice : dict
            SPICE configuration containing the embedder parameters.
        """
        # Forward the owned configuration block to the SPICE implementation
        super().__init__(**spice)


class SPICELoss(_SPICELoss):
    """Top-level SPICE loss exposed through the model registry."""


MODEL_SPEC = ModelSpec("spice", SPICE, SPICELoss)
