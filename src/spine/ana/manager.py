"""Manages the operation of analysis scripts."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from spine.utils.factory import parse_module_config
from spine.utils.manager import ModuleManager
from spine.utils.stopwatch import StopwatchManager

from .base import AnaBase
from .factories import ana_script_factory


class AnaManager(ModuleManager[AnaBase]):
    """Manager class to initialize and execute analysis scripts.

    Analysis scripts use the output of the reconstruction chain and the
    post-processors and produce simple CSV files.

    It loads all the analysis scripts and feeds them data. It initializes
    CSV writers needed to store the output of the analysis scripts.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        log_dir: str | None = None,
        prefix: str | None = None,
    ) -> None:
        """Initialize the analysis manager.

        Parameters
        ----------
        cfg : dict
            Analysis script configurations
        log_dir : str
            Output CSV file directory (shared with driver log)
        prefix : str, optional
            Input file prefix. If requested, it will be used to prefix
            all the output CSV files.
        """
        # Parse the analysis block configuration
        config = dict(cfg)

        overwrite = config.pop("overwrite", None)
        if overwrite is not None and not isinstance(overwrite, bool):
            raise TypeError("`overwrite` must be a boolean when provided.")

        prefix_output = config.pop("prefix_output", False)
        if not isinstance(prefix_output, bool):
            raise TypeError("`prefix_output` must be a boolean.")

        buffer_size = config.pop("buffer_size", 1)
        if not isinstance(buffer_size, int):
            raise TypeError("`buffer_size` must be an integer.")

        self.parse_config(
            log_dir,
            prefix,
            overwrite=overwrite,
            prefix_output=prefix_output,
            buffer_size=buffer_size,
            **config,
        )

    def parse_config(
        self,
        log_dir: str | None,
        prefix: str | None,
        overwrite: bool | None = None,
        prefix_output: bool = False,
        buffer_size: int = 1,
        **modules: dict[str, Any] | None,
    ) -> None:
        """Parse the analysis tool configuration.

        Parameters
        ----------
        log_dir : str
            Output CSV file directory (shared with driver log)
        prefix : str
            Input file prefix. If requested, it will be used to prefix
            all the output CSV files.
        overwrite : bool, optional
            If `True`, overwrite the CSV logs if they already exist
        prefix_output : bool, optional
            If `True`, will prefix the output CSV names with the input file name
        buffer_size : int, default 1
            CSV file buffer size. 1 is line buffered (default),
            -1 uses system default, 0 is unbuffered, >1 is buffer size in bytes
        **modules : dict
            List of analysis script modules
        """
        # Only use the prefix if the output is to be prefixed
        if not prefix_output:
            prefix = None

        # Add the modules to a processor list in decreasing order of priority
        self.watch = StopwatchManager()
        module_map: OrderedDict[str, AnaBase] = OrderedDict()
        parsed = parse_module_config(
            modules, sort_by_priority=True, priority_descending=True
        )
        for key, spec in parsed.items():
            # Profile the module
            self.watch.initialize(key)

            # Append
            module_map[key] = ana_script_factory(
                spec["name"],
                spec["cfg"],
                overwrite,
                log_dir,
                prefix,
                buffer_size,
            )
        self.modules = module_map

    def close(self) -> None:
        """Close all analysis script writers and flush remaining data.

        This should be called when analysis is complete to ensure all
        CSV files are properly closed and data is written.
        """
        for module in getattr(self, "modules", {}).values():
            module.close_writers()

    def flush(self) -> None:
        """Flush all analysis script writer buffers.

        This forces any buffered data to be written to disk without
        closing the files.
        """
        for module in self.modules.values():
            module.flush_writers()

    def __del__(self) -> None:
        """Destructor to ensure analysis writers are closed.

        Acts as a safety net in case close() is not called explicitly.
        """
        self.close()
