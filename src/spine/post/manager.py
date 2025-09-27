"""Manages the operation of post-processors."""

from collections import OrderedDict, defaultdict
from copy import deepcopy
from warnings import warn

import numpy as np

from spine.utils.stopwatch import StopwatchManager

from .factories import post_processor_factory


class PostManager:
    """Manager in charge of handling post-processing scripts.

    It loads all the post-processor objects once and feeds them data.
    """

    def __init__(self, cfg, post_list=None, parent_path=None):
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
        # Loop over the post-processor modules and get their priorities
        cfg = deepcopy(cfg)
        keys = np.array(list(cfg.keys()))
        priorities = -np.ones(len(keys), dtype=np.int32)
        for i, key in enumerate(keys):
            if "priority" in cfg[key]:
                priorities[i] = cfg[key].pop("priority")

        # Add the modules to a processor list in decreasing order of priority
        self.watch = StopwatchManager()
        self.modules = OrderedDict()
        keys = keys[np.argsort(-priorities)]
        for key in keys:
            # Profile the module
            self.watch.initialize(key)

            # Append
            self.modules[key] = post_processor_factory(
                key, cfg[key], parent_path=parent_path
            )

            # Check dependencies
            if post_list is not None:
                ups_post = tuple(self.modules)
                for post in self.modules[key]._upstream:
                    assert post in (post_list + ups_post), (
                        f"Post-processor `{key}` is missing an essential "
                        f"upstream post-processor: `{post}`."
                    )

    def __call__(self, data):
        """Pass one batch of data through the post-processors.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the post-processor modules
        single_entry = np.isscalar(data["index"])
        for key, module in self.modules.items():
            # Run the post-processor on each entry
            self.watch.start(key)
            if single_entry:
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
