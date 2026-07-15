"""Construct a post-processor module class from its name."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spine.utils.factory import instantiate, module_dict

from . import crt, optical, reco, trigger, truth

# Build a dictionary of available calibration modules
POST_DICT = {}
for module in [reco, truth, optical, crt, trigger]:
    POST_DICT.update(**module_dict(module))


def post_processor_factory(
    name: str, cfg: Mapping[str, Any], parent_path: str | None = None
) -> Any:
    """Instantiates a post-processor module from a configuration dictionary.

    Parameters
    ----------
    name : str
        Name of the post-processor module
    cfg : dict
        Post-processor module configuration
    parent_path : str, optional
        Path to the post-processor configuration file

    Returns
    -------
    object
         Initialized post-processor object
    """
    # Provide the name to the configuration
    config = dict(cfg)
    config["name"] = name

    # Instantiate the post-processor
    if POST_DICT[name].provide_parent_path:
        return instantiate(POST_DICT, config, parent_path=parent_path)
    else:
        return instantiate(POST_DICT, config)
