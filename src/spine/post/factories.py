"""Construct a post-processor module class from its name."""

from spine.utils.factory import instantiate, module_dict

from . import crt, optical, reco, trigger, truth

# Build a dictionary of available calibration modules
POST_DICT = {}
for module in [reco, truth, optical, crt, trigger]:
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
    cfg["name"] = name

    # Instantiate the post-processor module
    if name in POST_DICT and POST_DICT[name].need_parent_path:
        return instantiate(POST_DICT, cfg, parent_path=parent_path)
    else:
        return instantiate(POST_DICT, cfg)
