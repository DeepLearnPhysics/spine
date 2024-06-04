"""Construct a calibrator class from its name."""

from spine.utils.factory import module_dict, instantiate

from . import gain, lifetime, transparency, field, recombination


def calibrator_factory(name, cfg):
    """Instantiates calibrator module from a configuration dictionary.

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
    # Build a dictionary of available calibration modules
    calib_dict = {}
    for module in [gain, lifetime, transparency, recombination, field]:
        calib_dict.update(**module_dict(module))

    # Provide the name to the configuration
    cfg['name'] = name

    # Instantiate the calibration module
    return instantiate(calib_dict, cfg)
