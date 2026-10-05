"""Manages the operation of post-processors."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from spine.config.factory import parse_module_config
from spine.utils.manager import ModuleManager
from spine.utils.stopwatch import StopwatchManager

from .base import PostBase
from .factories import post_processor_factory


class PostManager(ModuleManager[PostBase]):
    """Manager in charge of handling post-processors.

    It loads all the post-processor objects once and feeds them data.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        post_list: Sequence[str] | None = None,
        parent_path: str | None = None,
    ) -> None:
        """Initialize the post-processing manager.

        Parameters
        ----------
        cfg : dict
            Legacy post-processor mappings or a ``stages`` list of named
            provider entries in execution order.
        post_list : sequence[str], optional
            List of post-processors which have already been run. ``None``
            preserves the legacy opt-out from dependency checking; an empty
            sequence checks dependencies against earlier stages only.
        parent_path : str, optional
            Path to the parent directory of the main configuration file
        """
        # Explicit stages retain list order; legacy blocks use descending priority.
        self.watch = StopwatchManager()
        modules: OrderedDict[str, PostBase] = OrderedDict()
        parsed = parse_module_config(
            cfg, sort_by_priority=True, priority_descending=True, stages_key="stages"
        )
        module_names: list[str] = []
        for key, spec in parsed.items():
            # Profile the module
            self.watch.initialize(key)

            # Construct before registering so a stage cannot satisfy its own
            # upstream dependency through its instance name.
            processor = post_processor_factory(
                spec["name"], spec["cfg"], parent_path=parent_path
            )

            # Check dependencies
            if post_list is not None:
                ups_post = tuple(post_list) + tuple(modules) + tuple(module_names)
                for post in processor._upstream:
                    if post not in ups_post:
                        raise ValueError(
                            f"Post-processor `{key}` is missing an essential "
                            f"upstream post-processor: `{post}`."
                        )
            modules[key] = processor
            module_names.append(spec["name"])

        self.modules = modules
        self.module_names = tuple(module_names)
