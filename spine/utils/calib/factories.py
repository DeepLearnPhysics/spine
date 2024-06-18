"""Construct a calibrator class from its name."""

from spine.utils.factory import module_dict, instantiate

from . import gain, lifetime, transparency, field, recombination

# Build a dictionary of available calibration modules
CALIB_DICT = {}
for module in [gain, lifetime, transparency, recombination, field]:
    CALIB_DICT.update(**module_dict(module))


def calibrator_factory(name, cfg):
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
    # Provide the name to the configuration
    cfg['name'] = name

    # Instantiate the calibration module
    return instantiate(CALIB_DICT, cfg)
