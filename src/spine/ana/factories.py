"""Construct an analysis script module class from its name."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spine.utils.factory import instantiate, module_dict

from . import calib, diag, metric, script

# Build a dictionary of available calibration modules
ANA_DICT = {}
for module in [calib, diag, metric, script]:
    ANA_DICT.update(**module_dict(module))


def ana_script_factory(
    name: str,
    cfg: Mapping[str, Any],
    overwrite: bool | None = None,
    log_dir: str | None = None,
    prefix: str | None = None,
    buffer_size: int = 1,
) -> Any:
    """Instantiates an analyzer module from a configuration dictionary.

    Parameters
    ----------
    name : str
        Name of the analyzer module
    cfg : dict
        Analysis script module configuration
    parent_path : str
        Path to the parent directory of the main analysis configuration. This
        allows for the use of relative paths in the analyzers.
    overwrite : bool, optional
        If `True`, overwrite the CSV logs if they already exist
    log_dir : str, optional
        Output CSV file directory (shared with driver log)
    prefix : str, optional
        Input file prefix. If requested, it will be used to prefix
        all the output CSV files.
    buffer_size : int, default 1
        CSV file buffer size for analysis outputs

    Returns
    -------
    object
         Initialized analyzer object
    """
    # Provide the name to the configuration
    config = dict(cfg)
    config["name"] = name

    # Instantiate the analysis script module
    if overwrite is not None:
        return instantiate(
            ANA_DICT,
            config,
            overwrite=overwrite,
            log_dir=log_dir,
            prefix=prefix,
            buffer_size=buffer_size,
        )
    else:
        return instantiate(
            ANA_DICT, config, log_dir=log_dir, prefix=prefix, buffer_size=buffer_size
        )
