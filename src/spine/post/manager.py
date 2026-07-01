"""Manages the operation of post-processors."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from spine.utils.factory import parse_module_config
from spine.utils.manager import ModuleManager
from spine.utils.stopwatch import StopwatchManager

from .factories import post_processor_factory


class PostModule(Protocol):
    """Post-processing module interface used by :class:`PostManager`."""

    _upstream: Sequence[str]

    def __call__(
        self, data: dict[str, Any], entry: int | None = None
    ) -> dict[str, Any] | None:
        """Process input data and return products to merge into it."""


class PostManager(ModuleManager[PostModule]):
    """Manager in charge of handling post-processing scripts.

    It loads all the post-processor objects once and feeds them data.
    """

    def __init__(
        self,
        cfg: Mapping[str, dict[str, Any] | None],
        post_list: Sequence[str] | None = None,
        parent_path: str | None = None,
    ) -> None:
        """Initialize the post-processing manager.

        Parameters
        ----------
        cfg : dict
            Post-processor configurations
        post_list : List[str], optional
            List of post-processors which have already been run
        parent_path : str, optional
            Path to the analysis tools configuration file
        """
        # Add the modules to a processor list in decreasing order of priority
        self.watch = StopwatchManager()
        modules: OrderedDict[str, PostModule] = OrderedDict()
        parsed = parse_module_config(
            cfg, sort_by_priority=True, priority_descending=True
        )
        module_names: list[str] = []
        for key, spec in parsed.items():
            # Profile the module
            self.watch.initialize(key)

            # Append
            modules[key] = post_processor_factory(
                spec["name"], spec["cfg"], parent_path=parent_path
            )

            # Check dependencies
            if post_list is not None:
                ups_post = tuple(post_list) + tuple(modules) + tuple(module_names)
                for post in modules[key]._upstream:
                    if post not in ups_post:
                        raise ValueError(
                            f"Post-processor `{key}` is missing an essential "
                            f"upstream post-processor: `{post}`."
                        )
            module_names.append(spec["name"])
        self.modules = modules
