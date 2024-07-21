"""Construct a post-processor module class from its name."""

from spine.utils.factory import module_dict, instantiate

from . import reco, metric, optical, crt, trigger

# Build a dictionary of available calibration modules
POST_DICT = {}
for module in [reco, metric, optical, crt, trigger]:
    POST_DICT.update(**module_dict(module))


def post_processor_factory(name, cfg, parent_path=None):
    """Instantiates a post-processor module from a configuration dictionary.

    Parameters
    ----------
    name : str
        Name of the post-processor module
    cfg : dict
        Post-processor module configuration

    Returns
    -------
    object
         Initialized post-processor object
    """
    # Provide the name to the configuration
    cfg['name'] = name

    # Instantiate the post-processor module
    # TODO: This is hacky, fix it
    if name == 'flash_match':
        return instantiate(POST_DICT, cfg, parent_path=parent_path)
    else:
        return instantiate(POST_DICT, cfg)
