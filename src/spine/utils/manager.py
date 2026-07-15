"""Shared manager utilities for configurable per-entry modules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from spine.utils.stopwatch import StopwatchManager


class ManagedModule(Protocol):
    """Callable module which can process a batch or a single batch entry."""

    def __call__(
        self, data: dict[str, Any], entry: int | None = None
    ) -> dict[str, Any] | None:
        """Process input data and return products to merge into it."""


ModuleT = TypeVar("ModuleT", bound=ManagedModule)


class ModuleManager(Generic[ModuleT]):
    """Base class for managers that run ordered modules on event data."""

    modules: Mapping[str, ModuleT]
    watch: StopwatchManager

    def __call__(self, data: dict[str, Any]) -> None:
        """Pass one batch of data through the managed modules.

        Parameters
        ----------
        data : dict
            Dictionary of data products. It is updated in place.
        """
        # Reset active stopwatches
        self.watch.reset_if_active()

        # Loop over the managed modules
        single_entry = np.isscalar(data["index"])
        for module_key, module in self.modules.items():
            # Run the module on each entry
            self.watch.start(module_key)
            if single_entry:
                num_entries = 1
                result = module(data)

            else:
                num_entries = len(data["index"])
                result = defaultdict(list)
                for entry in range(num_entries):
                    result_e = module(data, entry)
                    if result_e is not None:
                        for key, value in result_e.items():
                            result[key].append(value)

            self.watch.stop(module_key)

            # Update the input dictionary
            if result is not None:
                for key, value in result.items():
                    if not single_entry and len(value) != num_entries:
                        raise ValueError(
                            f"Module `{module_key}` returned {len(value)} values "
                            f"for `{key}`, but the batch contains {num_entries} "
                            "entries."
                        )
                    data[key] = value
