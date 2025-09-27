"""Construct a analysis script module class from its name."""

from spine.utils.factory import instantiate, module_dict

from . import calib, diag, metric, script

# Build a dictionary of available calibration modules
ANA_DICT = {}
for module in [calib, diag, metric, script]:
    ANA_DICT.update(**module_dict(module))


def ana_script_factory(name, cfg, overwrite=None, log_dir=None, prefix=None):
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

    Returns
    -------
    object
         Initialized analyzer object
    """
    # Provide the name to the configuration
    cfg["name"] = name

    # Instantiate the analysis script module
    if overwrite is not None:
        return instantiate(
            ANA_DICT, cfg, overwrite=overwrite, log_dir=log_dir, prefix=prefix
        )
    else:
        return instantiate(ANA_DICT, cfg, log_dir=log_dir, prefix=prefix)
