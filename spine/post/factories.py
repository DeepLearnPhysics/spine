"""Construct a post-processor module class from its name."""

from spine.utils.factory import module_dict, instantiate

from . import reco, metric, optical, crt, trigger

# Build a dictionary of available calibration modules
POST_DICT = {}
for module in [reco, metric, optical, crt, trigger]:
    POST_DICT.update(**module_dict(module))


def post_processor_factory(name, cfg, parent_path=''):
    """Instantiates a post-processor module from a configuration dictionary.

    Parameters
    ----------
    name : str
        Name of the post-processor module
    cfg : dict
        Post-processor module configuration
    parent_path : str
        Path to the parent directory of the main analysis configuration. This
        allows for the use of relative paths in the post-processors.

    Returns
    -------
    object
         Initialized post-processor object
    """
    # Provide the name to the configuration
    cfg['name'] = name

    # Instantiate the post-processor module
    return instantiate(POST_DICT, cfg)
