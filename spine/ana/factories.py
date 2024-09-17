"""Construct a analysis script module class from its name."""

from spine.utils.factory import module_dict, instantiate

from . import metric, script

# Build a dictionary of available calibration modules
ANA_DICT = {}
for module in [metric, script]:
    ANA_DICT.update(**module_dict(module))


def ana_script_factory(name, cfg, parent_path='', overwrite=False):
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
    overwrite : bool, default False
        If `True`, overwrite the CSV logs if they already exist

    Returns
    -------
    object
         Initialized analyzer object
    """
    # Provide the name to the configuration
    cfg['name'] = name

    # Instantiate the analysis script module
    return instantiate(ANA_DICT, cfg, overwrite=overwrite)
