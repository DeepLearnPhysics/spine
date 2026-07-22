"""Construct a calibrator class from its name."""

from __future__ import annotations

from typing import Any

from spine.utils.factory import instantiate, module_dict

from . import field, gain, lifetime, recombination, response, transparency

# Build a dictionary of available calibration modules
CALIB_DICT: dict[str, type] = {}
for module in [gain, lifetime, transparency, response, recombination, field]:
    CALIB_DICT.update(**module_dict(module))


def calibrator_factory(name: str, cfg: dict[str, Any]) -> object:
    """Instantiates a calibrator module from a configuration dictionary.

    Parameters
    ----------
    name : str
        Name of the calibration module
    cfg : dict
        Calibration module configuration

    Returns
    -------
    object
         Initialized calibration module
    """
    # Instantiate the calibration module
    cfg = dict(cfg)
    cfg["name"] = name
    return instantiate(CALIB_DICT, cfg)
