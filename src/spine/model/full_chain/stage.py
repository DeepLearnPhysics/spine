"""Stage and loss contracts for full-chain providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .state import ChainState, StageResult

__all__ = ["ChainStage", "ChainLossStage"]


class ChainStage(ABC):
    """Adapt one reconstruction implementation to canonical chain products.

    Subclasses declare their product contract through ``requires``,
    ``optional``, ``provides`` and ``replaces``. The orchestrator validates
    that contract before registering any execution plan.
    """

    requires: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    replaces: frozenset[str] = frozenset()

    def __init__(self, name: str) -> None:
        """Initialize a named stage instance.

        Parameters
        ----------
        name : str
            Unique name of this stage within the execution plan.
        """
        self.name = name

    def validate(self, available: set[str]) -> set[str]:
        """Validate dependencies and return products available afterward.

        Parameters
        ----------
        available : set of str
            Canonical products available before this stage runs.

        Returns
        -------
        set of str
            Canonical products available after this stage runs.
        """
        missing = self.requires.difference(available)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"Stage `{self.name}` requires unavailable products: {names}."
            )

        collisions = self.provides.intersection(available).difference(self.replaces)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(
                f"Stage `{self.name}` would replace undeclared products: {names}."
            )
        return available.union(self.provides)

    def __call__(self, state: ChainState) -> StageResult:
        """Validate runtime inputs and execute the provider.

        Parameters
        ----------
        state : ChainState
            Canonical products and public outputs accumulated so far.

        Returns
        -------
        StageResult
            Products and outputs published by this stage.
        """
        for key in self.requires:
            state.require(key, self.name)
        return self.forward(state)

    @abstractmethod
    def forward(self, state: ChainState) -> StageResult:
        """Execute the stage and return canonical and public outputs.

        Parameters
        ----------
        state : ChainState
            Current chain execution state.

        Returns
        -------
        StageResult
            Products and outputs produced by the implementation.
        """


class ChainLossStage(ABC):
    """Adapt one standalone objective to full-chain products and outputs.

    Loss stages translate stable public chain outputs back into the native
    argument names expected by their standalone objective implementations.
    """

    def __init__(self, name: str) -> None:
        """Initialize a named loss stage.

        Parameters
        ----------
        name : str
            Name of the corresponding network stage.
        """
        self.name = name

    @abstractmethod
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the stage objective.

        Parameters
        ----------
        data : dict
            Driver truth products and public network outputs.

        Returns
        -------
        dict
            Native objective loss and diagnostic metrics.
        """

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the stage objective.

        Parameters
        ----------
        data : dict
            Driver truth products and public network outputs.

        Returns
        -------
        dict
            Native objective loss and diagnostic metrics.
        """
        return self.forward(data)
