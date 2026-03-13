"""Manages the operation of analysis scripts."""

from collections import OrderedDict, defaultdict
from copy import deepcopy

import numpy as np

from spine.utils.stopwatch import StopwatchManager

from .factories import ana_script_factory


class AnaManager:
    """Manager class to initialize and execute analysis scripts.

    Analysis scripts use the output of the reconstruction chain and the
    post-processors and produce simple CSV files.

    It loads all the analysis scripts and feeds them data. It initializes
    CSV writers needed to store the output of the analysis scripts.
    """

    def __init__(self, cfg, log_dir=None, prefix=None):
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
        self.parse_config(log_dir, prefix, **cfg)

    def parse_config(
        self,
        log_dir,
        prefix,
        overwrite=None,
        prefix_output=False,
        buffer_size=1,
        **modules,
    ):
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
        # Loop over the analyzer modules and get their priorities
        modules = deepcopy(modules)
        keys = np.array(list(modules.keys()))
        priorities = -np.ones(len(keys), dtype=np.int32)
        for i, k in enumerate(keys):
            if "priority" in modules[k]:
                priorities[i] = modules[k].pop("priority")

        # Only use the prefix if the output is to be prefixed
        if not prefix_output:
            prefix = None

        # Add the modules to a processor list in decreasing order of priority
        self.watch = StopwatchManager()
        self.modules = OrderedDict()
        keys = keys[np.argsort(-priorities)]
        for k in keys:
            # Profile the module
            self.watch.initialize(k)

            # Append
            self.modules[k] = ana_script_factory(
                k, modules[k], overwrite, log_dir, prefix, buffer_size
            )

    def __call__(self, data):
        """Pass one batch of data through the analysis scripts

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the analysis script modules
        single_entry = np.isscalar(data["index"])
        for key, module in self.modules.items():
            # Run the analysis script on each entry
            self.watch.start(key)
            if single_entry:
                num_entries = 1
                result = module(data)

            else:
                num_entries = len(data["index"])
                result = defaultdict(list)
                for entry in range(num_entries):
                    result_e = module(data, entry)
                    if result_e is not None:
                        for k, v in result_e.items():
                            result[k].append(v)

            self.watch.stop(key)

            # Update the input dictionary
            if result is not None:
                for key, val in result.items():
                    if not single_entry:
                        assert len(val) == num_entries, (
                            f"The number {key} ({len(val)}) does not match "
                            f"the number of entries ({num_entries})."
                        )
                    data[key] = val

    def close(self):
        """Close all analysis script writers and flush remaining data.

        This should be called when analysis is complete to ensure all
        CSV files are properly closed and data is written.
        """
        for module in self.modules.values():
            module.close_writers()

    def flush(self):
        """Flush all analysis script writer buffers.

        This forces any buffered data to be written to disk without
        closing the files.
        """
        for module in self.modules.values():
            module.flush_writers()

    def __del__(self):
        """Destructor to ensure analysis writers are closed.

        Acts as a safety net in case close() is not called explicitly.
        """
        self.close()
